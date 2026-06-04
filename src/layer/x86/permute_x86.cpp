// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "permute_x86.h"

#include <string.h>
#include <vector>

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

namespace {

static void transpose_pack1_fp32(const float* const* rows, int row_count, int col_count, float* outptr)
{
    int i = 0;

#if __SSE2__
#if __AVX512F__
    for (; i + 15 < row_count; i += 16)
    {
        int j = 0;
        for (; j + 15 < col_count; j += 16)
        {
            __m512 _r0 = _mm512_loadu_ps(rows[i] + j);
            __m512 _r1 = _mm512_loadu_ps(rows[i + 1] + j);
            __m512 _r2 = _mm512_loadu_ps(rows[i + 2] + j);
            __m512 _r3 = _mm512_loadu_ps(rows[i + 3] + j);
            __m512 _r4 = _mm512_loadu_ps(rows[i + 4] + j);
            __m512 _r5 = _mm512_loadu_ps(rows[i + 5] + j);
            __m512 _r6 = _mm512_loadu_ps(rows[i + 6] + j);
            __m512 _r7 = _mm512_loadu_ps(rows[i + 7] + j);
            __m512 _r8 = _mm512_loadu_ps(rows[i + 8] + j);
            __m512 _r9 = _mm512_loadu_ps(rows[i + 9] + j);
            __m512 _ra = _mm512_loadu_ps(rows[i + 10] + j);
            __m512 _rb = _mm512_loadu_ps(rows[i + 11] + j);
            __m512 _rc = _mm512_loadu_ps(rows[i + 12] + j);
            __m512 _rd = _mm512_loadu_ps(rows[i + 13] + j);
            __m512 _re = _mm512_loadu_ps(rows[i + 14] + j);
            __m512 _rf = _mm512_loadu_ps(rows[i + 15] + j);

            transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

            _mm512_storeu_ps(outptr + (j + 0) * row_count + i, _r0);
            _mm512_storeu_ps(outptr + (j + 1) * row_count + i, _r1);
            _mm512_storeu_ps(outptr + (j + 2) * row_count + i, _r2);
            _mm512_storeu_ps(outptr + (j + 3) * row_count + i, _r3);
            _mm512_storeu_ps(outptr + (j + 4) * row_count + i, _r4);
            _mm512_storeu_ps(outptr + (j + 5) * row_count + i, _r5);
            _mm512_storeu_ps(outptr + (j + 6) * row_count + i, _r6);
            _mm512_storeu_ps(outptr + (j + 7) * row_count + i, _r7);
            _mm512_storeu_ps(outptr + (j + 8) * row_count + i, _r8);
            _mm512_storeu_ps(outptr + (j + 9) * row_count + i, _r9);
            _mm512_storeu_ps(outptr + (j + 10) * row_count + i, _ra);
            _mm512_storeu_ps(outptr + (j + 11) * row_count + i, _rb);
            _mm512_storeu_ps(outptr + (j + 12) * row_count + i, _rc);
            _mm512_storeu_ps(outptr + (j + 13) * row_count + i, _rd);
            _mm512_storeu_ps(outptr + (j + 14) * row_count + i, _re);
            _mm512_storeu_ps(outptr + (j + 15) * row_count + i, _rf);
        }

        for (; j < col_count; j++)
        {
            for (int k = 0; k < 16; k++)
            {
                outptr[j * row_count + i + k] = rows[i + k][j];
            }
        }
    }
#endif // __AVX512F__

#if __AVX__
    for (; i + 7 < row_count; i += 8)
    {
        int j = 0;
        for (; j + 7 < col_count; j += 8)
        {
            __m256 _r0 = _mm256_loadu_ps(rows[i] + j);
            __m256 _r1 = _mm256_loadu_ps(rows[i + 1] + j);
            __m256 _r2 = _mm256_loadu_ps(rows[i + 2] + j);
            __m256 _r3 = _mm256_loadu_ps(rows[i + 3] + j);
            __m256 _r4 = _mm256_loadu_ps(rows[i + 4] + j);
            __m256 _r5 = _mm256_loadu_ps(rows[i + 5] + j);
            __m256 _r6 = _mm256_loadu_ps(rows[i + 6] + j);
            __m256 _r7 = _mm256_loadu_ps(rows[i + 7] + j);

            transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

            _mm256_storeu_ps(outptr + (j + 0) * row_count + i, _r0);
            _mm256_storeu_ps(outptr + (j + 1) * row_count + i, _r1);
            _mm256_storeu_ps(outptr + (j + 2) * row_count + i, _r2);
            _mm256_storeu_ps(outptr + (j + 3) * row_count + i, _r3);
            _mm256_storeu_ps(outptr + (j + 4) * row_count + i, _r4);
            _mm256_storeu_ps(outptr + (j + 5) * row_count + i, _r5);
            _mm256_storeu_ps(outptr + (j + 6) * row_count + i, _r6);
            _mm256_storeu_ps(outptr + (j + 7) * row_count + i, _r7);
        }

        for (; j < col_count; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                outptr[j * row_count + i + k] = rows[i + k][j];
            }
        }
    }
#endif // __AVX__

    for (; i + 3 < row_count; i += 4)
    {
        int j = 0;
        for (; j + 3 < col_count; j += 4)
        {
            __m128 _r0 = _mm_loadu_ps(rows[i] + j);
            __m128 _r1 = _mm_loadu_ps(rows[i + 1] + j);
            __m128 _r2 = _mm_loadu_ps(rows[i + 2] + j);
            __m128 _r3 = _mm_loadu_ps(rows[i + 3] + j);

            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

            _mm_storeu_ps(outptr + (j + 0) * row_count + i, _r0);
            _mm_storeu_ps(outptr + (j + 1) * row_count + i, _r1);
            _mm_storeu_ps(outptr + (j + 2) * row_count + i, _r2);
            _mm_storeu_ps(outptr + (j + 3) * row_count + i, _r3);
        }

        for (; j < col_count; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                outptr[j * row_count + i + k] = rows[i + k][j];
            }
        }
    }
#endif // __SSE2__

    for (; i < row_count; i++)
    {
        for (int j = 0; j < col_count; j++)
        {
            outptr[j * row_count + i] = rows[i][j];
        }
    }
}

static void transpose_contiguous_pack1_fp32(const float* ptr, int w, int h, float* outptr)
{
    std::vector<const float*> rows(h);
    for (int i = 0; i < h; i++)
    {
        rows[i] = ptr + (size_t)i * w;
    }

    transpose_pack1_fp32(rows.data(), h, w, outptr);
}

} // namespace

Permute_x86::Permute_x86()
{
#if __SSE2__
    support_packing = true;
#endif
}

static void unpack_permute_repack(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    Mat bottom_blob_unpacked;
    {
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;
        ncnn::convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_unpack);
    }

    Mat top_unpacked;
    Permute permute_op;
    permute_op.order_type = order_type;
    permute_op.forward(bottom_blob_unpacked, top_unpacked, opt);

    if (bottom_blob.elempack > 1 && top_unpacked.c % bottom_blob.elempack == 0)
    {
        ncnn::convert_packing(top_unpacked, top_blob, bottom_blob.elempack, opt);
    }
    else
    {
        top_blob = top_unpacked;
    }
}

int Permute_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const size_t elemsize = bottom_blob.elemsize;
    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const bool use_fp32_pack1_fast_path = bottom_blob.elembits() == 32 && elempack == 1;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (elempack > 1 && bottom_blob.elembits() == 32)
    {
        // packed fp32: unpack → pack1 permute → repack
        unpack_permute_repack(bottom_blob, top_blob, order_type, opt);
        return 0;
    }

    if (dims == 2)
    {
        if (order_type == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (order_type == 1)
        {
            top_blob.create(h, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (use_fp32_pack1_fast_path)
            {
                transpose_contiguous_pack1_fp32(bottom_blob, w, h, top_blob);
                return 0;
            }

            unsigned char* outptr = top_blob;

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    const unsigned char* ptr = bottom_blob.row<const unsigned char>(j) + (size_t)i * elemsize;
                    memcpy(outptr, ptr, elemsize);
                    outptr += elemsize;
                }
            }

            return 0;
        }
    }

    if (dims == 3)
    {
        if (order_type == 1)
        {
            top_blob.create(h, w, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (use_fp32_pack1_fast_path)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    transpose_contiguous_pack1_fp32(bottom_blob.channel(q), w, h, top_blob.channel(q));
                }

                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                unsigned char* outptr = top_blob.channel(q);

                for (int i = 0; i < w; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        const unsigned char* ptr = m.row<const unsigned char>(j) + (size_t)i * elemsize;
                        memcpy(outptr, ptr, elemsize);
                        outptr += elemsize;
                    }
                }
            }

            return 0;
        }

        if (order_type == 2)
        {
            top_blob.create(w, channels, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int i = 0; i < channels; i++)
                {
                    const unsigned char* ptr = bottom_blob.channel(i).row<const unsigned char>(q);
                    memcpy(outptr, ptr, (size_t)w * elemsize);
                    outptr += (size_t)w * elemsize;
                }
            }

            return 0;
        }

        if (order_type == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (order_type == 3)
        {
            top_blob.create(channels, w, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (use_fp32_pack1_fast_path)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < h; q++)
                {
                    float* outptr = top_blob.channel(q);
                    std::vector<const float*> rows(channels);

                    for (int j = 0; j < channels; j++)
                    {
                        rows[j] = bottom_blob.channel(j).row(q);
                    }

                    transpose_pack1_fp32(rows.data(), channels, w, outptr);
                }

                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int i = 0; i < w; i++)
                {
                    for (int j = 0; j < channels; j++)
                    {
                        const unsigned char* ptr = bottom_blob.channel(j).row<const unsigned char>(q) + (size_t)i * elemsize;
                        memcpy(outptr, ptr, elemsize);
                        outptr += elemsize;
                    }
                }
            }

            return 0;
        }
        if (order_type == 4)
        {
            top_blob.create(h, channels, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int i = 0; i < channels; i++)
                {
                    const Mat m = bottom_blob.channel(i);

                    for (int j = 0; j < h; j++)
                    {
                        const unsigned char* ptr = m.row<const unsigned char>(j) + (size_t)q * elemsize;
                        memcpy(outptr, ptr, elemsize);
                        outptr += elemsize;
                    }
                }
            }

            return 0;
        }
        if (order_type == 5)
        {
            top_blob.create(channels, h, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < channels; j++)
                    {
                        const unsigned char* ptr = bottom_blob.channel(j).row<const unsigned char>(i) + (size_t)q * elemsize;
                        memcpy(outptr, ptr, elemsize);
                        outptr += elemsize;
                    }
                }
            }

            return 0;
        }
    }

    if (dims == 4)
    {
        if (order_type == 0)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (order_type == 1)
        {
            top_blob.create(h, w, d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (use_fp32_pack1_fast_path)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat in_channel = bottom_blob.channel(q);
                    Mat out_channel = top_blob.channel(q);

                    for (int z = 0; z < d; z++)
                    {
                        transpose_contiguous_pack1_fp32(in_channel.depth(z), w, h, out_channel.depth(z));
                    }
                }

                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    const Mat m = bottom_blob.channel(q).depth(z);

                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < h; j++)
                        {
                            const unsigned char* ptr = m.row<const unsigned char>(j) + (size_t)i * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }

        if (order_type == 2)
        {
            top_blob.create(w, d, h, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        const unsigned char* ptr = bottom_blob.channel(q).depth(i).row<const unsigned char>(z);
                        memcpy(outptr, ptr, (size_t)w * elemsize);
                        outptr += (size_t)w * elemsize;
                    }
                }
            }

            return 0;
        }

        if (order_type == 3)
        {
            top_blob.create(d, w, h, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (use_fp32_pack1_fast_path)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    float* outptr = top_blob.channel(q);
                    std::vector<const float*> rows(d);

                    for (int y = 0; y < h; y++)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            rows[z] = m.depth(z).row(y);
                        }

                        transpose_pack1_fp32(rows.data(), d, w, outptr);
                        outptr += (size_t)w * d;
                    }
                }

                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            const unsigned char* ptr = m.depth(j).row<const unsigned char>(z) + (size_t)i * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }

        if (order_type == 4)
        {
            top_blob.create(h, d, w, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        const Mat m = bottom_blob.channel(q).depth(i);

                        for (int j = 0; j < h; j++)
                        {
                            const unsigned char* ptr = m.row<const unsigned char>(j) + (size_t)z * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 5)
        {
            top_blob.create(d, h, w, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            const unsigned char* ptr = m.depth(j).row<const unsigned char>(i) + (size_t)z * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 6)
        {
            top_blob.create(w, h, channels, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        const unsigned char* ptr = bottom_blob.channel(z).depth(q).row<const unsigned char>(i);
                        memcpy(outptr, ptr, (size_t)w * elemsize);
                        outptr += (size_t)w * elemsize;
                    }
                }
            }

            return 0;
        }
        if (order_type == 7)
        {
            top_blob.create(h, w, channels, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (use_fp32_pack1_fast_path)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < d; q++)
                {
                    Mat out_channel = top_blob.channel(q);

                    for (int z = 0; z < channels; z++)
                    {
                        transpose_contiguous_pack1_fp32(bottom_blob.channel(z).depth(q), w, h, out_channel.channel(z));
                    }
                }

                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    const Mat m = bottom_blob.channel(z).depth(q);

                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < h; j++)
                        {
                            const unsigned char* ptr = m.row<const unsigned char>(j) + (size_t)i * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 8)
        {
            top_blob.create(w, channels, h, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const unsigned char* ptr = bottom_blob.channel(i).depth(q).row<const unsigned char>(z);
                        memcpy(outptr, ptr, (size_t)w * elemsize);
                        outptr += (size_t)w * elemsize;
                    }
                }
            }

            return 0;
        }
        if (order_type == 9)
        {
            top_blob.create(channels, w, h, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (use_fp32_pack1_fast_path)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < d; q++)
                {
                    float* outptr = top_blob.channel(q);

                    for (int z = 0; z < h; z++)
                    {
                        for (int i = 0; i < w; i++)
                        {
                            for (int j = 0; j < channels; j++)
                            {
                                *outptr++ = bottom_blob.channel(j).depth(q).row(z)[i];
                            }
                        }
                    }
                }

                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            const unsigned char* ptr = bottom_blob.channel(j).depth(q).row<const unsigned char>(z) + (size_t)i * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 10)
        {
            top_blob.create(h, channels, w, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const Mat m = bottom_blob.channel(i).depth(q);

                        for (int j = 0; j < h; j++)
                        {
                            const unsigned char* ptr = m.row<const unsigned char>(j) + (size_t)z * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 11)
        {
            top_blob.create(channels, h, w, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            const unsigned char* ptr = bottom_blob.channel(j).depth(q).row<const unsigned char>(i) + (size_t)z * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 12)
        {
            top_blob.create(w, d, channels, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        const unsigned char* ptr = bottom_blob.channel(z).depth(i).row<const unsigned char>(q);
                        memcpy(outptr, ptr, (size_t)w * elemsize);
                        outptr += (size_t)w * elemsize;
                    }
                }
            }

            return 0;
        }
        if (order_type == 13)
        {
            top_blob.create(d, w, channels, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (use_fp32_pack1_fast_path)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < h; q++)
                {
                    Mat out_channel = top_blob.channel(q);
                    std::vector<const float*> rows(d);

                    for (int z = 0; z < channels; z++)
                    {
                        float* outptr = out_channel.channel(z);

                        for (int j = 0; j < d; j++)
                        {
                            rows[j] = bottom_blob.channel(z).depth(j).row(q);
                        }

                        transpose_pack1_fp32(rows.data(), d, w, outptr);
                    }
                }

                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    const Mat m = bottom_blob.channel(z);

                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            const unsigned char* ptr = m.depth(j).row<const unsigned char>(q) + (size_t)i * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 14)
        {
            top_blob.create(w, channels, d, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const unsigned char* ptr = bottom_blob.channel(i).depth(z).row<const unsigned char>(q);
                        memcpy(outptr, ptr, (size_t)w * elemsize);
                        outptr += (size_t)w * elemsize;
                    }
                }
            }

            return 0;
        }
        if (order_type == 15)
        {
            top_blob.create(channels, w, d, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (use_fp32_pack1_fast_path)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < h; q++)
                {
                    Mat out_channel = top_blob.channel(q);
                    std::vector<const float*> rows(channels);

                    for (int z = 0; z < d; z++)
                    {
                        float* outptr = out_channel.channel(z);

                        for (int j = 0; j < channels; j++)
                        {
                            rows[j] = bottom_blob.channel(j).depth(z).row(q);
                        }

                        transpose_pack1_fp32(rows.data(), channels, w, outptr);
                    }
                }

                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            const unsigned char* ptr = bottom_blob.channel(j).depth(z).row<const unsigned char>(q) + (size_t)i * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 16)
        {
            top_blob.create(d, channels, w, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const Mat m = bottom_blob.channel(i);

                        for (int j = 0; j < d; j++)
                        {
                            const unsigned char* ptr = m.depth(j).row<const unsigned char>(q) + (size_t)z * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 17)
        {
            top_blob.create(channels, d, w, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            const unsigned char* ptr = bottom_blob.channel(j).depth(i).row<const unsigned char>(q) + (size_t)z * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 18)
        {
            top_blob.create(h, d, channels, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        const Mat m = bottom_blob.channel(z).depth(i);

                        for (int j = 0; j < h; j++)
                        {
                            const unsigned char* ptr = m.row<const unsigned char>(j) + (size_t)q * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 19)
        {
            top_blob.create(d, h, channels, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        const Mat m = bottom_blob.channel(z);

                        for (int j = 0; j < d; j++)
                        {
                            const unsigned char* ptr = m.depth(j).row<const unsigned char>(i) + (size_t)q * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 20)
        {
            top_blob.create(h, channels, d, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const Mat m = bottom_blob.channel(i).depth(z);

                        for (int j = 0; j < h; j++)
                        {
                            const unsigned char* ptr = m.row<const unsigned char>(j) + (size_t)q * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 21)
        {
            top_blob.create(channels, h, d, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            const unsigned char* ptr = bottom_blob.channel(j).depth(z).row<const unsigned char>(i) + (size_t)q * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 22)
        {
            top_blob.create(d, channels, h, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const Mat m = bottom_blob.channel(i);

                        for (int j = 0; j < d; j++)
                        {
                            const unsigned char* ptr = m.depth(j).row<const unsigned char>(z) + (size_t)q * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
        if (order_type == 23)
        {
            top_blob.create(channels, d, h, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                unsigned char* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            const unsigned char* ptr = bottom_blob.channel(j).depth(i).row<const unsigned char>(z) + (size_t)q * elemsize;
                            memcpy(outptr, ptr, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }

            return 0;
        }
    }

    return 0;
}

} // namespace ncnn
