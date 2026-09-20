// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "permute_x86.h"

#include <limits.h>
#include <string.h>

#include "cpu.h"
#include "x86_usability.h"

namespace ncnn {

#include "permute_fp32.h"
#include "permute_bf16s_fp16s.h"

Permute_x86::Permute_x86()
{
#if __SSE2__
    support_packing = true;
    support_any_packing = true;
#endif // __SSE2__
    support_fp16_storage = cpu_support_x86_f16c();
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

// split only when the existing channel/slice tasks cannot occupy the threads
// keep each task large enough to amortize scheduling and align output segments
static int permute_block_size(int size, size_t bytes_per_element, int groups, int num_threads, int alignment)
{
    if (num_threads <= 1 || groups >= num_threads)
        return size;

    const int blocks = (num_threads + groups - 1) / groups;
    const int min_block = (int)std::min((size_t)size, (16384 + bytes_per_element - 1) / bytes_per_element);
    int block = std::max((size + blocks - 1) / blocks, min_block);
    block = (block + alignment - 1) / alignment * alignment;
    return std::min(block, size);
}

// complete matrix traversal
// each output row contains rows contiguous blocks
// serial tile kernels never create OpenMP teams
static void permute_transpose_matrix(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int size, int num_threads)
{
    if (num_threads == 1)
    {
        permute_transpose_blocks(ptr, stride, outptr, outstride, rows, cols, size);
        return;
    }

    // a few long contiguous blocks can be split without changing their order
    if ((size_t)rows * cols <= (size_t)num_threads && size * sizeof(float) >= 16384)
    {
        const int block = permute_block_size(size, sizeof(float), rows * cols, num_threads, 64 / sizeof(float));
        #pragma omp parallel for collapse(3) num_threads(num_threads)
        for (int j = 0; j < cols; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int x = 0; x < size; x += block)
                {
                    const float* p = ptr + i * stride + (size_t)j * size + x;
                    float* out = outptr + j * outstride + (size_t)i * size + x;
                    memcpy(out, p, (size_t)std::min(block, size - x) * sizeof(float));
                }
            }
        }
        return;
    }

    const int col_block = permute_block_size(cols, (size_t)rows * size * sizeof(float), 1, num_threads, 4);
    const int row_block = permute_block_size(rows, (size_t)col_block * size * sizeof(float), (cols + col_block - 1) / col_block, num_threads, 4);
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int j = 0; j < cols; j += col_block)
    {
        for (int i = 0; i < rows; i += row_block)
        {
            permute_transpose_blocks(ptr + i * stride + (size_t)j * size, stride, outptr + j * outstride + (size_t)i * size, outstride, std::min(row_block, rows - i), std::min(col_block, cols - j), size);
        }
    }
}

// complete matrix traversal
// each output row contains rows contiguous blocks
// serial tile kernels never create OpenMP teams
static void permute_transpose_matrix_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int size, int num_threads)
{
    if (num_threads == 1)
    {
        permute_transpose_blocks_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols, size);
        return;
    }

    // a few long contiguous blocks can be split without changing their order
    if ((size_t)rows * cols <= (size_t)num_threads && size * sizeof(unsigned short) >= 16384)
    {
        const int block = permute_block_size(size, sizeof(unsigned short), rows * cols, num_threads, 64 / sizeof(unsigned short));
        #pragma omp parallel for collapse(3) num_threads(num_threads)
        for (int j = 0; j < cols; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int x = 0; x < size; x += block)
                {
                    const unsigned short* p = ptr + i * stride + (size_t)j * size + x;
                    unsigned short* out = outptr + j * outstride + (size_t)i * size + x;
                    memcpy(out, p, (size_t)std::min(block, size - x) * sizeof(unsigned short));
                }
            }
        }
        return;
    }

    const int col_block = permute_block_size(cols, (size_t)rows * size * sizeof(unsigned short), 1, num_threads, 4);
    const int row_block = permute_block_size(rows, (size_t)col_block * size * sizeof(unsigned short), (cols + col_block - 1) / col_block, num_threads, 4);
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int j = 0; j < cols; j += col_block)
    {
        for (int i = 0; i < rows; i += row_block)
        {
            permute_transpose_blocks_bf16s_fp16s(ptr + i * stride + (size_t)j * size, stride, outptr + j * outstride + (size_t)i * size, outstride, std::min(row_block, rows - i), std::min(col_block, cols - j), size);
        }
    }
}

int Permute_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elembits() == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);

    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
#ifdef _OPENMP
    const int num_threads = bottom_blob.total() * elemsize >= 65536 ? opt.num_threads : 1;
#else
    const int num_threads = 1;
#endif

    if (dims == 1 || order_type == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (bottom_blob.elembits() != 32)
        return -100;

    if (dims == 2)
    {
        // order_type
        // 0 = w h
        // 1 = h w

        if (order_type == 1)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h * elempack, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                permute_transpose_matrix(bottom_blob, w, top_blob, h, h, w, 1, num_threads);
                return 0;
            }

            const int row_block = permute_block_size(h, (size_t)w * elemsize, (w + 31) / 32, num_threads, 32);
            #pragma omp parallel for collapse(2) num_threads(num_threads)
            for (int i = 0; i < h; i += row_block)
            {
                for (int x = 0; x < w; x += 32)
                {
                    const float* ptr = (const float*)bottom_blob + (size_t)i * w * elempack + x * elempack;
                    float* outptr = top_blob.row<float>(x / out_elempack) + (size_t)i * elempack * out_elempack;
                    permute_transpose2d(ptr, (size_t)w * elempack, outptr, (size_t)top_blob.w * out_elempack, std::min(row_block, h - i) * elempack, std::min(32, w - x), elempack, out_elempack);
                }
            }
            return 0;
        }
    }

    if (dims == 3)
    {
        // order_type
        // 0 = w h c
        // 1 = h w c
        // 2 = w c h
        // 3 = c w h
        // 4 = h c w
        // 5 = c h w

        if (order_type == 1)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, w, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (num_threads == 1)
            {
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_blob.channel(q);
                    float* outptr = (float*)top_blob.channel(q * elempack / out_elempack);
                    permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                }
                return 0;
            }
            if (channels >= num_threads)
            {
                #pragma omp parallel for num_threads(num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_blob.channel(q);
                    float* outptr = (float*)top_blob.channel(q * elempack / out_elempack);
                    permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                }
                return 0;
            }

            const int row_block = permute_block_size(h, (size_t)w * elemsize, channels, num_threads, 32);
            const int col_block = permute_block_size(w, (size_t)h * elemsize, channels * ((h + row_block - 1) / row_block), num_threads, 32);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        const float* ptr = (const float*)bottom_blob.channel(q) + i * ((size_t)w * elempack) + j * elempack;
                        float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + j * ((size_t)h * out_elempack) + i * out_elempack;
                        permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, std::min(row_block, h - i), std::min(col_block, w - j), elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 2)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, channels * elempack, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                if (w < 32)
                {
                    // short blocks include the w == 1 matrix-transpose case
                    permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, top_blob.cstep, channels, h, w, num_threads);
                    return 0;
                }

                const int block = permute_block_size(w, sizeof(float), (top_blob.c) * (top_blob.h), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < w; i += block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {

                        for (int y = 0; y < top_blob.h; y++)
                        {
                            const float* ptr = (const float*)bottom_blob + q * (size_t)w + y * bottom_blob.cstep;
                            float* outptr = (float*)top_blob.channel(q) + (size_t)y * w;
                            memcpy(outptr + i, ptr + i, (size_t)std::min(block, w - i) * sizeof(float));
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, top_blob.c * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                        float* outptr = (float*)top_blob.channel(q) + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                        // exchange c and h, keeping w as the inner spatial axis
                        permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                  elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 3)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, w, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                // w and h stay adjacent in the output; retain input channel padding
                if (top_blob.cstep == (size_t)w * channels && (size_t)w * h <= INT_MAX)
                {
                    permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, channels, channels, w * h, 1, num_threads);
                    return 0;
                }

                const int col_block = permute_block_size(w, (size_t)channels * sizeof(float), h, num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(float), h * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int y = 0; y < h; y++)
                        {
                            const float* ptr = (const float*)bottom_blob + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                            float* outptr = (float*)top_blob + y * top_blob.cstep + (size_t)j * ((size_t)top_blob.w) + i;
                            permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w, std::min(row_block, channels - i), std::min(col_block, w - j));
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, top_blob.c * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                        float* outptr = (float*)top_blob.channel(q) + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * out_elempack);
                        // exchange c and h, keeping w as the inner spatial axis
                        permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                  elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 4)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, channels * elempack, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                // c and h stay adjacent in the input; retain output channel padding
                if (bottom_blob.cstep == (size_t)w * h && (size_t)channels * h <= INT_MAX)
                {
                    permute_transpose_matrix(bottom_blob, w, top_blob, top_blob.cstep, channels * h, w, 1, num_threads);
                    return 0;
                }

                const int col_block = permute_block_size(w, (size_t)h * sizeof(float), channels, num_threads, 32);
                const int row_block = permute_block_size(h, (size_t)w * sizeof(float), channels * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < h; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + (size_t)i * ((size_t)w) + j;
                            float* outptr = (float*)top_blob + q * (size_t)top_blob.w + (size_t)j * (top_blob.cstep) + i;
                            permute_transpose_pack1(ptr, (size_t)w, outptr, top_blob.cstep, std::min(row_block, h - i), std::min(col_block, w - j));
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, top_blob.c * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                        float* outptr = (float*)top_blob.channel(q) + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                        // exchange c and w, keeping h as the inner spatial axis
                        permute3d(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                  (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 5)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, h, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                // with w or h removed, the remaining operation is one c/spatial transpose
                if (w == 1 || h == 1)
                {
                    permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, w == 1 ? (size_t)channels : top_blob.cstep, channels, w * h, 1, num_threads);
                    return 0;
                }

                if (channels == 1)
                {
                    permute_transpose_matrix(bottom_blob, w, top_blob, top_blob.cstep, h, w, 1, num_threads);
                    return 0;
                }

                const int col_block = permute_block_size(w, (size_t)channels * sizeof(float), h, num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(float), h * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int y = 0; y < h; y++)
                        {
                            const float* ptr = (const float*)bottom_blob + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                            float* outptr = (float*)top_blob + y * (size_t)top_blob.w + (size_t)j * (top_blob.cstep) + i;
                            permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, top_blob.cstep, std::min(row_block, channels - i), std::min(col_block, w - j));
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, top_blob.c * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                        float* outptr = (float*)top_blob.channel(q) + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * out_elempack);
                        // exchange c and w, keeping h as the inner spatial axis
                        permute3d(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                  (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }
    }

    if (dims == 4)
    {
        // order_type
        // 0 = w h d c
        // 1 = h w d c
        // 2 = w d h c
        // 3 = d w h c
        // 4 = h d w c
        // 5 = d h w c
        // 6 = w h c d
        // 7 = h w c d
        // 8 = w c h d
        // 9 = c w h d
        // 10 = h c w d
        // 11 = c h w d
        // 12 = w d c h
        // 13 = d w c h
        // 14 = w c d h
        // 15 = c w d h
        // 16 = d c w h
        // 17 = c d w h
        // 18 = h d c w
        // 19 = d h c w
        // 20 = h c d w
        // 21 = c h d w
        // 22 = d c h w
        // 23 = c d h w

        if (order_type == 1)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, w, d, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (num_threads == 1)
            {
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)z * w * h * elempack;
                        float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * w * h * out_elempack;
                        permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    }
                }
                return 0;
            }
            if (channels * d >= num_threads)
            {
                #pragma omp parallel for collapse(2) num_threads(num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)z * w * h * elempack;
                        float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * w * h * out_elempack;
                        permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    }
                }
                return 0;
            }

            const int row_block = permute_block_size(h, (size_t)w * elemsize, channels * d, num_threads, 32);
            const int col_block = permute_block_size(w, (size_t)h * elemsize, channels * d * ((h + row_block - 1) / row_block), num_threads, 32);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i += row_block)
                    {
                        for (int j = 0; j < w; j += col_block)
                        {
                            const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)z * w * h * elempack + i * ((size_t)w * elempack) + j * elempack;
                            float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * w * h * out_elempack + j * ((size_t)h * out_elempack) + i * out_elempack;
                            permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, std::min(row_block, h - i), std::min(col_block, w - j), elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 2)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, d, h, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == out_elempack)
            {
                if (num_threads == 1)
                {
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = bottom_blob.channel(q);
                        float* outptr = top_blob.channel(q);
                        permute_transpose_blocks(ptr, (size_t)w * h * elempack, outptr, (size_t)w * d * elempack, d, h, w * elempack);
                    }
                    return 0;
                }
                if (channels >= num_threads)
                {
                    #pragma omp parallel for num_threads(num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = bottom_blob.channel(q);
                        float* outptr = top_blob.channel(q);
                        permute_transpose_blocks(ptr, (size_t)w * h * elempack, outptr, (size_t)w * d * elempack, d, h, w * elempack);
                    }
                }
                else if (channels == 1)
                {
                    permute_transpose_matrix(bottom_blob, (size_t)w * h * elempack, top_blob, (size_t)w * d * elempack, d, h, w * elempack, num_threads);
                }
                else
                {
                    const int col_block = permute_block_size(h, (size_t)d * (w * elempack) * sizeof(float), channels, num_threads, 4);
                    const int row_block = permute_block_size(d, (size_t)col_block * (w * elempack) * sizeof(float), channels * ((h + col_block - 1) / col_block), num_threads, 64 / sizeof(float));
                    #pragma omp parallel for collapse(3) num_threads(num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        for (int j = 0; j < h; j += col_block)
                        {
                            for (int i = 0; i < d; i += row_block)
                            {
                                const float* ptr = (const float*)bottom_blob.channel(q) + i * ((size_t)w * h * elempack) + (size_t)j * (w * elempack);
                                float* outptr = (float*)top_blob.channel(q) + j * ((size_t)w * d * elempack) + (size_t)i * (w * elempack);
                                permute_transpose_blocks(ptr, (size_t)w * h * elempack, outptr, (size_t)w * d * elempack, std::min(row_block, d - i), std::min(col_block, h - j), w * elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, elemsize, channels * h * d, num_threads, 32);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int i = 0; i < w; i += block)
                        {
                            const float* ptr = (const float*)bottom_blob.channel(q) + ((size_t)z * h + y) * w * elempack + i * elempack;
                            float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (((size_t)y * d + z) * w + i) * out_elempack;
                            permute_unpack_spatial(ptr, outptr, top_blob.cstep, std::min(block, w - i), elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 3)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, w, h, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (num_threads == 1)
            {
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)y * w * elempack;
                        float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * w * d * out_elempack;
                        permute_transpose_spatial(ptr, (size_t)w * h * elempack, outptr, (size_t)d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    }
                }
                return 0;
            }
            if (channels * h >= num_threads)
            {
                #pragma omp parallel for collapse(2) num_threads(num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)y * w * elempack;
                        float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * w * d * out_elempack;
                        permute_transpose_spatial(ptr, (size_t)w * h * elempack, outptr, (size_t)d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    }
                }
                return 0;
            }

            const int row_block = permute_block_size(d, (size_t)w * elemsize, channels * h, num_threads, 32);
            const int col_block = permute_block_size(w, (size_t)d * elemsize, channels * h * ((d + row_block - 1) / row_block), num_threads, 32);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int i = 0; i < d; i += row_block)
                    {
                        for (int j = 0; j < w; j += col_block)
                        {
                            const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)y * w * elempack + i * ((size_t)w * h * elempack) + j * elempack;
                            float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * w * d * out_elempack + j * ((size_t)d * out_elempack) + i * out_elempack;
                            permute_transpose_spatial(ptr, (size_t)w * h * elempack, outptr, (size_t)d * out_elempack, top_blob.cstep, std::min(row_block, d - i), std::min(col_block, w - j), elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 4)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, d, w, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (num_threads == 1)
            {
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)z * w * h * elempack;
                        float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * h * out_elempack;
                        permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    }
                }
                return 0;
            }
            if (channels * d >= num_threads)
            {
                #pragma omp parallel for collapse(2) num_threads(num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)z * w * h * elempack;
                        float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * h * out_elempack;
                        permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    }
                }
                return 0;
            }

            const int row_block = permute_block_size(h, (size_t)w * elemsize, channels * d, num_threads, 32);
            const int col_block = permute_block_size(w, (size_t)h * elemsize, channels * d * ((h + row_block - 1) / row_block), num_threads, 32);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i += row_block)
                    {
                        for (int j = 0; j < w; j += col_block)
                        {
                            const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)z * w * h * elempack + i * ((size_t)w * elempack) + j * elempack;
                            float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * h * out_elempack + j * ((size_t)h * d * out_elempack) + i * out_elempack;
                            permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, std::min(row_block, h - i), std::min(col_block, w - j), elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 5)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, h, w, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (num_threads == 1)
            {
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)y * w * elempack;
                        float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * d * out_elempack;
                        permute_transpose_spatial(ptr, (size_t)w * h * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    }
                }
                return 0;
            }
            if (channels * h >= num_threads)
            {
                #pragma omp parallel for collapse(2) num_threads(num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)y * w * elempack;
                        float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * d * out_elempack;
                        permute_transpose_spatial(ptr, (size_t)w * h * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    }
                }
                return 0;
            }

            const int row_block = permute_block_size(d, (size_t)w * elemsize, channels * h, num_threads, 32);
            const int col_block = permute_block_size(w, (size_t)d * elemsize, channels * h * ((d + row_block - 1) / row_block), num_threads, 32);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int i = 0; i < d; i += row_block)
                    {
                        for (int j = 0; j < w; j += col_block)
                        {
                            const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)y * w * elempack + i * ((size_t)w * h * elempack) + j * elempack;
                            float* outptr = (float*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * d * out_elempack + j * ((size_t)h * d * out_elempack) + i * out_elempack;
                            permute_transpose_spatial(ptr, (size_t)w * h * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, std::min(row_block, d - i), std::min(col_block, w - j), elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 6)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, h, channels * elempack, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                // w and h stay adjacent on both sides
                permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, top_blob.cstep, channels, d, w * h, num_threads);
                return 0;
            }

            // w and h stay adjacent on both sides
            const int block = permute_block_size(w * h, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)(w * h) * elempack * out_elemsize, top_blob.c * ((w * h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < w * h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                        float* outptr = (float*)top_blob.channel(q) + (size_t)c * elempack * ((size_t)w * h * out_elempack) + i * out_elempack;
                        permute3d(ptr, outptr, std::min(block, w * h - i), std::min(channel_block, channels - c),
                                  elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)w * h * out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 7)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, w, channels * elempack, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)h * sizeof(float), (channels) * (d), num_threads, 32);
                const int row_block = permute_block_size(h, (size_t)w * sizeof(float), ((channels) * (d)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < h; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int z = 0; z < d; z++)
                            {
                                const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h + (size_t)i * ((size_t)w) + j;
                                float* outptr = (float*)top_blob + q * (size_t)top_blob.w * top_blob.h + z * top_blob.cstep + (size_t)j * ((size_t)top_blob.w) + i;
                                permute_transpose_pack1(ptr, (size_t)w, outptr, (size_t)top_blob.w, std::min(row_block, h - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (w) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((w) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < h; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const float* ptr = (const float*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                                float* outptr = (float*)top_blob + x * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * out_elempack;
                                // exchange c and d, keeping h as the inner spatial axis
                                permute3d(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                          (size_t)w * elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((h) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int y = 0; y < h; y++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            float* outptr = (float*)top_blob + y * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * ((size_t)top_blob.w * out_elempack);
                            // exchange c and d, keeping w as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                      elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 8)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, channels * elempack, h, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                if (num_threads == 1)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = (const float*)bottom_blob + (size_t)z * w * h;
                        float* outptr = (float*)top_blob + z * top_blob.cstep;
                        permute_transpose_blocks(ptr, bottom_blob.cstep, outptr, (size_t)channels * w, channels, h, w);
                    }
                    return 0;
                }
                if (d >= num_threads)
                {
                    #pragma omp parallel for num_threads(num_threads)
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = (const float*)bottom_blob + (size_t)z * w * h;
                        float* outptr = (float*)top_blob + z * top_blob.cstep;
                        permute_transpose_blocks(ptr, bottom_blob.cstep, outptr, (size_t)channels * w, channels, h, w);
                    }
                }
                else if (d == 1)
                {
                    permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, (size_t)channels * w, channels, h, w, num_threads);
                }
                else
                {
                    const int col_block = permute_block_size(h, (size_t)channels * (w) * sizeof(float), d, num_threads, 4);
                    const int row_block = permute_block_size(channels, (size_t)col_block * (w) * sizeof(float), d * ((h + col_block - 1) / col_block), num_threads, 64 / sizeof(float));
                    #pragma omp parallel for collapse(3) num_threads(num_threads)
                    for (int z = 0; z < d; z++)
                    {
                        for (int j = 0; j < h; j += col_block)
                        {
                            for (int i = 0; i < channels; i += row_block)
                            {
                                const float* ptr = (const float*)bottom_blob + (size_t)z * w * h + i * (bottom_blob.cstep) + (size_t)j * (w);
                                float* outptr = (float*)top_blob + z * top_blob.cstep + j * ((size_t)channels * w) + (size_t)i * (w);
                                permute_transpose_blocks(ptr, bottom_blob.cstep, outptr, (size_t)channels * w, std::min(row_block, channels - i), std::min(col_block, h - j), w);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((h) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int y = 0; y < h; y++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            float* outptr = (float*)top_blob + y * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                            // exchange c and d, keeping w as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                      elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 9)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, w, h, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w * h, (size_t)channels * sizeof(float), d, num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)(w * h) * sizeof(float), d * ((w * h + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w * h; j += col_block)
                    {
                        for (int q = 0; q < d; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + (size_t)q * w * h + (size_t)i * (bottom_blob.cstep) + j;
                            float* outptr = (float*)top_blob.channel(q) + (size_t)j * (channels) + i;
                            permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, channels, std::min(row_block, channels - i), std::min(col_block, w * h - j));
                        }
                    }
                }
                return 0;
            }

            // w and h stay adjacent on both sides
            const int block = permute_block_size(w * h, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)(w * h) * elempack * out_elemsize, top_blob.c * ((w * h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < w * h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                        float* outptr = (float*)top_blob.channel(q) + (size_t)c * elempack * out_elempack + i * ((size_t)channels * elempack * out_elempack);
                        permute3d(ptr, outptr, std::min(block, w * h - i), std::min(channel_block, channels - c),
                                  elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, (size_t)channels * elempack * out_elempack, out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 10)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, channels * elempack, w, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)h * sizeof(float), (channels) * (d), num_threads, 32);
                const int row_block = permute_block_size(h, (size_t)w * sizeof(float), ((channels) * (d)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < h; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int z = 0; z < d; z++)
                            {
                                const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h + (size_t)i * ((size_t)w) + j;
                                float* outptr = (float*)top_blob + q * (size_t)top_blob.w + z * top_blob.cstep + (size_t)j * ((size_t)top_blob.w * top_blob.h) + i;
                                permute_transpose_pack1(ptr, (size_t)w, outptr, (size_t)top_blob.w * top_blob.h, std::min(row_block, h - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (w) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((w) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < h; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const float* ptr = (const float*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                                float* outptr = (float*)top_blob + x * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                                // exchange c and d, keeping h as the inner spatial axis
                                permute3d(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                          (size_t)w * elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((h) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int y = 0; y < h; y++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            float* outptr = (float*)top_blob + y * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and d, keeping w as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                      elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 11)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, h, w, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)channels * sizeof(float), (d) * (h), num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(float), ((d) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const float* ptr = (const float*)bottom_blob + z * (size_t)w * h + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                                float* outptr = (float*)top_blob + z * top_blob.cstep + y * (size_t)top_blob.w + (size_t)j * ((size_t)top_blob.w * top_blob.h) + i;
                                permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w * top_blob.h, std::min(row_block, channels - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((h) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int y = 0; y < h; y++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            float* outptr = (float*)top_blob + y * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and d, keeping w as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                      elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 12)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, d, channels * elempack, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                // c and d have the same order on both sides; merge only without input padding
                if (bottom_blob.cstep == (size_t)w * h * d && (size_t)channels * d <= INT_MAX)
                {
                    permute_transpose_matrix(bottom_blob, (size_t)w * h, top_blob, top_blob.cstep, channels * d, h, w, num_threads);
                    return 0;
                }

                if (num_threads == 1)
                {
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep;
                        float* outptr = (float*)top_blob + (size_t)q * d * w;
                        permute_transpose_blocks(ptr, (size_t)w * h, outptr, top_blob.cstep, d, h, w);
                    }
                    return 0;
                }
                if (channels >= num_threads)
                {
                    #pragma omp parallel for num_threads(num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep;
                        float* outptr = (float*)top_blob + (size_t)q * d * w;
                        permute_transpose_blocks(ptr, (size_t)w * h, outptr, top_blob.cstep, d, h, w);
                    }
                }
                else if (channels == 1)
                {
                    permute_transpose_matrix(bottom_blob, (size_t)w * h, top_blob, top_blob.cstep, d, h, w, num_threads);
                }
                else
                {
                    const int col_block = permute_block_size(h, (size_t)d * (w) * sizeof(float), channels, num_threads, 4);
                    const int row_block = permute_block_size(d, (size_t)col_block * (w) * sizeof(float), channels * ((h + col_block - 1) / col_block), num_threads, 64 / sizeof(float));
                    #pragma omp parallel for collapse(3) num_threads(num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        for (int j = 0; j < h; j += col_block)
                        {
                            for (int i = 0; i < d; i += row_block)
                            {
                                const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + i * ((size_t)w * h) + (size_t)j * (w);
                                float* outptr = (float*)top_blob + (size_t)q * d * w + j * (top_blob.cstep) + (size_t)i * (w);
                                permute_transpose_blocks(ptr, (size_t)w * h, outptr, top_blob.cstep, std::min(row_block, d - i), std::min(col_block, h - j), w);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            float* outptr = (float*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * out_elempack;
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                      elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 13)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, w, channels * elempack, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)d * sizeof(float), (channels) * (h), num_threads, 32);
                const int row_block = permute_block_size(d, (size_t)w * sizeof(float), ((channels) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w + (size_t)i * ((size_t)w * h) + j;
                                float* outptr = (float*)top_blob + q * (size_t)top_blob.w * top_blob.h + y * top_blob.cstep + (size_t)j * ((size_t)top_blob.w) + i;
                                permute_transpose_pack1(ptr, (size_t)w * h, outptr, (size_t)top_blob.w, std::min(row_block, d - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize, (w) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize, ((w) * (top_blob.c)) * ((d + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const float* ptr = (const float*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * h * elempack);
                                float* outptr = (float*)top_blob + x * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * out_elempack;
                                // exchange c and h, keeping d as the inner spatial axis
                                permute3d(ptr, outptr, std::min(block, d - i), std::min(channel_block, channels - c),
                                          (size_t)w * h * elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            float* outptr = (float*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * ((size_t)top_blob.w * out_elempack);
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                      elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 14)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, channels * elempack, d, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                if (num_threads == 1)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = (const float*)bottom_blob + (size_t)z * w * h;
                        float* outptr = (float*)top_blob + (size_t)z * channels * w;
                        permute_transpose_blocks(ptr, bottom_blob.cstep, outptr, top_blob.cstep, channels, h, w);
                    }
                    return 0;
                }
                if (d >= num_threads)
                {
                    #pragma omp parallel for num_threads(num_threads)
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = (const float*)bottom_blob + (size_t)z * w * h;
                        float* outptr = (float*)top_blob + (size_t)z * channels * w;
                        permute_transpose_blocks(ptr, bottom_blob.cstep, outptr, top_blob.cstep, channels, h, w);
                    }
                }
                else if (d == 1)
                {
                    permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, top_blob.cstep, channels, h, w, num_threads);
                }
                else
                {
                    const int col_block = permute_block_size(h, (size_t)channels * (w) * sizeof(float), d, num_threads, 4);
                    const int row_block = permute_block_size(channels, (size_t)col_block * (w) * sizeof(float), d * ((h + col_block - 1) / col_block), num_threads, 64 / sizeof(float));
                    #pragma omp parallel for collapse(3) num_threads(num_threads)
                    for (int z = 0; z < d; z++)
                    {
                        for (int j = 0; j < h; j += col_block)
                        {
                            for (int i = 0; i < channels; i += row_block)
                            {
                                const float* ptr = (const float*)bottom_blob + (size_t)z * w * h + i * (bottom_blob.cstep) + (size_t)j * (w);
                                float* outptr = (float*)top_blob + (size_t)z * channels * w + j * (top_blob.cstep) + (size_t)i * (w);
                                permute_transpose_blocks(ptr, bottom_blob.cstep, outptr, top_blob.cstep, std::min(row_block, channels - i), std::min(col_block, h - j), w);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            float* outptr = (float*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                      elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 15)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, w, d, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)channels * sizeof(float), (d) * (h), num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(float), ((d) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const float* ptr = (const float*)bottom_blob + z * (size_t)w * h + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                                float* outptr = (float*)top_blob + z * (size_t)top_blob.w * top_blob.h + y * top_blob.cstep + (size_t)j * ((size_t)top_blob.w) + i;
                                permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w, std::min(row_block, channels - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            float* outptr = (float*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * out_elempack);
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                      elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 16)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, channels * elempack, w, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)d * sizeof(float), (channels) * (h), num_threads, 32);
                const int row_block = permute_block_size(d, (size_t)w * sizeof(float), ((channels) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w + (size_t)i * ((size_t)w * h) + j;
                                float* outptr = (float*)top_blob + q * (size_t)top_blob.w + y * top_blob.cstep + (size_t)j * ((size_t)top_blob.w * top_blob.h) + i;
                                permute_transpose_pack1(ptr, (size_t)w * h, outptr, (size_t)top_blob.w * top_blob.h, std::min(row_block, d - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize, (w) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize, ((w) * (top_blob.c)) * ((d + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const float* ptr = (const float*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * h * elempack);
                                float* outptr = (float*)top_blob + x * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                                // exchange c and h, keeping d as the inner spatial axis
                                permute3d(ptr, outptr, std::min(block, d - i), std::min(channel_block, channels - c),
                                          (size_t)w * h * elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            float* outptr = (float*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                      elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 17)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, d, w, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)channels * sizeof(float), (d) * (h), num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(float), ((d) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const float* ptr = (const float*)bottom_blob + z * (size_t)w * h + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                                float* outptr = (float*)top_blob + z * (size_t)top_blob.w + y * top_blob.cstep + (size_t)j * ((size_t)top_blob.w * top_blob.h) + i;
                                permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w * top_blob.h, std::min(row_block, channels - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            float* outptr = (float*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                      elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 18)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, d, channels * elempack, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)(h * d) * sizeof(float), channels, num_threads, 32);
                const int row_block = permute_block_size(h * d, (size_t)w * sizeof(float), channels * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < h * d; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            const float* ptr = (const float*)bottom_blob.channel(q) + (size_t)i * (w) + j;
                            float* outptr = (float*)top_blob + (size_t)q * h * d + (size_t)j * (top_blob.cstep) + i;
                            permute_transpose_pack1(ptr, w, outptr, top_blob.cstep, std::min(row_block, h * d - i), std::min(col_block, w - j));
                        }
                    }
                }
                return 0;
            }

            // h and d stay adjacent on both sides
            const int block = permute_block_size(h * d, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)(h * d) * elempack * out_elemsize, top_blob.c * ((h * d + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < h * d; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                        float* outptr = (float*)top_blob.channel(q) + (size_t)c * elempack * ((size_t)h * d * out_elempack) + i * out_elempack;
                        permute3d(ptr, outptr, std::min(block, h * d - i), std::min(channel_block, channels - c),
                                  (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)h * d * out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 19)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, h, channels * elempack, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)d * sizeof(float), (channels) * (h), num_threads, 32);
                const int row_block = permute_block_size(d, (size_t)w * sizeof(float), ((channels) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w + (size_t)i * ((size_t)w * h) + j;
                                float* outptr = (float*)top_blob + q * (size_t)top_blob.w * top_blob.h + y * (size_t)top_blob.w + (size_t)j * (top_blob.cstep) + i;
                                permute_transpose_pack1(ptr, (size_t)w * h, outptr, top_blob.cstep, std::min(row_block, d - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize, ((h) * (top_blob.c)) * ((d + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int y = 0; y < h; y++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * h * elempack);
                                float* outptr = (float*)top_blob + y * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * out_elempack;
                                // exchange c and w, keeping d as the inner spatial axis
                                permute3d(ptr, outptr, std::min(block, d - i), std::min(channel_block, channels - c),
                                          (size_t)w * h * elempack, elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((d) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                            float* outptr = (float*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * ((size_t)top_blob.w * out_elempack);
                            // exchange c and w, keeping h as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                      (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 20)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, channels * elempack, d, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)h * sizeof(float), (channels) * (d), num_threads, 32);
                const int row_block = permute_block_size(h, (size_t)w * sizeof(float), ((channels) * (d)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < h; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int z = 0; z < d; z++)
                            {
                                const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h + (size_t)i * ((size_t)w) + j;
                                float* outptr = (float*)top_blob + q * (size_t)top_blob.w + z * (size_t)top_blob.w * top_blob.h + (size_t)j * (top_blob.cstep) + i;
                                permute_transpose_pack1(ptr, (size_t)w, outptr, top_blob.cstep, std::min(row_block, h - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((d) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                            float* outptr = (float*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                            // exchange c and w, keeping h as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                      (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 21)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, h, d, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)channels * sizeof(float), (d) * (h), num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(float), ((d) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const float* ptr = (const float*)bottom_blob + z * (size_t)w * h + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                                float* outptr = (float*)top_blob + z * (size_t)top_blob.w * top_blob.h + y * (size_t)top_blob.w + (size_t)j * (top_blob.cstep) + i;
                                permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, top_blob.cstep, std::min(row_block, channels - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            // h and d stay adjacent on both sides
            const int block = permute_block_size(h * d, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)(h * d) * elempack * out_elemsize, top_blob.c * ((h * d + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < h * d; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                        float* outptr = (float*)top_blob.channel(q) + (size_t)c * elempack * out_elempack + i * ((size_t)channels * elempack * out_elempack);
                        permute3d(ptr, outptr, std::min(block, h * d - i), std::min(channel_block, channels - c),
                                  (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, (size_t)channels * elempack * out_elempack, out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 22)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, channels * elempack, h, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)d * sizeof(float), (channels) * (h), num_threads, 32);
                const int row_block = permute_block_size(d, (size_t)w * sizeof(float), ((channels) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w + (size_t)i * ((size_t)w * h) + j;
                                float* outptr = (float*)top_blob + q * (size_t)top_blob.w + y * (size_t)top_blob.w * top_blob.h + (size_t)j * (top_blob.cstep) + i;
                                permute_transpose_pack1(ptr, (size_t)w * h, outptr, top_blob.cstep, std::min(row_block, d - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize, ((h) * (top_blob.c)) * ((d + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int y = 0; y < h; y++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * h * elempack);
                                float* outptr = (float*)top_blob + y * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                                // exchange c and w, keeping d as the inner spatial axis
                                permute3d(ptr, outptr, std::min(block, d - i), std::min(channel_block, channels - c),
                                          (size_t)w * h * elempack, elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((d) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                            float* outptr = (float*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and w, keeping h as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                      (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 23)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, d, h, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)channels * sizeof(float), (d) * (h), num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(float), ((d) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const float* ptr = (const float*)bottom_blob + z * (size_t)w * h + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                                float* outptr = (float*)top_blob + z * (size_t)top_blob.w + y * (size_t)top_blob.w * top_blob.h + (size_t)j * (top_blob.cstep) + i;
                                permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, top_blob.cstep, std::min(row_block, channels - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((d) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                            float* outptr = (float*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and w, keeping h as the inner spatial axis
                            permute3d(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                      (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }
    }

    return -1;
}

int Permute_x86::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
#ifdef _OPENMP
    const int num_threads = bottom_blob.total() * elemsize >= 65536 ? opt.num_threads : 1;
#else
    const int num_threads = 1;
#endif

    if (dims == 1 || order_type == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (bottom_blob.elembits() != 16)
        return -100;

    if (dims == 2)
    {
        // order_type
        // 0 = w h
        // 1 = h w

        if (order_type == 1)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h * elempack, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                permute_transpose_matrix_bf16s_fp16s(bottom_blob, w, top_blob, h, h, w, 1, num_threads);
                return 0;
            }

            const int row_block = permute_block_size(h, (size_t)w * elemsize, (w + 31) / 32, num_threads, 32);
            #pragma omp parallel for collapse(2) num_threads(num_threads)
            for (int i = 0; i < h; i += row_block)
            {
                for (int x = 0; x < w; x += 32)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)i * w * elempack + x * elempack;
                    unsigned short* outptr = top_blob.row<unsigned short>(x / out_elempack) + (size_t)i * elempack * out_elempack;
                    permute_transpose2d_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)top_blob.w * out_elempack, std::min(row_block, h - i) * elempack, std::min(32, w - x), elempack, out_elempack);
                }
            }
            return 0;
        }
    }

    if (dims == 3)
    {
        // order_type
        // 0 = w h c
        // 1 = h w c
        // 2 = w c h
        // 3 = c w h
        // 4 = h c w
        // 5 = c h w

        if (order_type == 1)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, w, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (num_threads == 1)
            {
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q);
                    unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack);
                    permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                }
                return 0;
            }
            if (channels >= num_threads)
            {
                #pragma omp parallel for num_threads(num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q);
                    unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack);
                    permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                }
                return 0;
            }

            const int row_block = permute_block_size(h, (size_t)w * elemsize, channels, num_threads, 32);
            const int col_block = permute_block_size(w, (size_t)h * elemsize, channels * ((h + row_block - 1) / row_block), num_threads, 32);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + i * ((size_t)w * elempack) + j * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + j * ((size_t)h * out_elempack) + i * out_elempack;
                        permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, std::min(row_block, h - i), std::min(col_block, w - j), elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 2)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, channels * elempack, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                if (w < 64)
                {
                    // short blocks include the w == 1 matrix-transpose case
                    permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, top_blob.cstep, channels, h, w, num_threads);
                    return 0;
                }

                const int block = permute_block_size(w, sizeof(unsigned short), (top_blob.c) * (top_blob.h), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < w; i += block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {

                        for (int y = 0; y < top_blob.h; y++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + q * (size_t)w + y * bottom_blob.cstep;
                            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + (size_t)y * w;
                            memcpy(outptr + i, ptr + i, (size_t)std::min(block, w - i) * sizeof(unsigned short));
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, top_blob.c * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                        // exchange c and h, keeping w as the inner spatial axis
                        permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                              elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 3)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, w, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                // w and h stay adjacent in the output; retain input channel padding
                if (top_blob.cstep == (size_t)w * channels && (size_t)w * h <= INT_MAX)
                {
                    permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, channels, channels, w * h, 1, num_threads);
                    return 0;
                }

                const int col_block = permute_block_size(w, (size_t)channels * sizeof(unsigned short), h, num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(unsigned short), h * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int y = 0; y < h; y++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                            unsigned short* outptr = (unsigned short*)top_blob + y * top_blob.cstep + (size_t)j * ((size_t)top_blob.w) + i;
                            permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w, std::min(row_block, channels - i), std::min(col_block, w - j));
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, top_blob.c * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * out_elempack);
                        // exchange c and h, keeping w as the inner spatial axis
                        permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                              elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 4)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, channels * elempack, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                // c and h stay adjacent in the input; retain output channel padding
                if (bottom_blob.cstep == (size_t)w * h && (size_t)channels * h <= INT_MAX)
                {
                    permute_transpose_matrix_bf16s_fp16s(bottom_blob, w, top_blob, top_blob.cstep, channels * h, w, 1, num_threads);
                    return 0;
                }

                const int col_block = permute_block_size(w, (size_t)h * sizeof(unsigned short), channels, num_threads, 32);
                const int row_block = permute_block_size(h, (size_t)w * sizeof(unsigned short), channels * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < h; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + (size_t)i * ((size_t)w) + j;
                            unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w + (size_t)j * (top_blob.cstep) + i;
                            permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w, outptr, top_blob.cstep, std::min(row_block, h - i), std::min(col_block, w - j));
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, top_blob.c * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                        // exchange c and w, keeping h as the inner spatial axis
                        permute3d_bf16s_fp16s(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                              (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 5)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, h, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                // with w or h removed, the remaining operation is one c/spatial transpose
                if (w == 1 || h == 1)
                {
                    permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, w == 1 ? (size_t)channels : top_blob.cstep, channels, w * h, 1, num_threads);
                    return 0;
                }

                if (channels == 1)
                {
                    permute_transpose_matrix_bf16s_fp16s(bottom_blob, w, top_blob, top_blob.cstep, h, w, 1, num_threads);
                    return 0;
                }

                const int col_block = permute_block_size(w, (size_t)channels * sizeof(unsigned short), h, num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(unsigned short), h * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int y = 0; y < h; y++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                            unsigned short* outptr = (unsigned short*)top_blob + y * (size_t)top_blob.w + (size_t)j * (top_blob.cstep) + i;
                            permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, top_blob.cstep, std::min(row_block, channels - i), std::min(col_block, w - j));
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, top_blob.c * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * out_elempack);
                        // exchange c and w, keeping h as the inner spatial axis
                        permute3d_bf16s_fp16s(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                              (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }
    }

    if (dims == 4)
    {
        // order_type
        // 0 = w h d c
        // 1 = h w d c
        // 2 = w d h c
        // 3 = d w h c
        // 4 = h d w c
        // 5 = d h w c
        // 6 = w h c d
        // 7 = h w c d
        // 8 = w c h d
        // 9 = c w h d
        // 10 = h c w d
        // 11 = c h w d
        // 12 = w d c h
        // 13 = d w c h
        // 14 = w c d h
        // 15 = c w d h
        // 16 = d c w h
        // 17 = c d w h
        // 18 = h d c w
        // 19 = d h c w
        // 20 = h c d w
        // 21 = c h d w
        // 22 = d c h w
        // 23 = c d h w

        if (order_type == 1)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, w, d, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (num_threads == 1)
            {
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)z * w * h * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * w * h * out_elempack;
                        permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    }
                }
                return 0;
            }
            if (channels * d >= num_threads)
            {
                #pragma omp parallel for collapse(2) num_threads(num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)z * w * h * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * w * h * out_elempack;
                        permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    }
                }
                return 0;
            }

            const int row_block = permute_block_size(h, (size_t)w * elemsize, channels * d, num_threads, 32);
            const int col_block = permute_block_size(w, (size_t)h * elemsize, channels * d * ((h + row_block - 1) / row_block), num_threads, 32);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i += row_block)
                    {
                        for (int j = 0; j < w; j += col_block)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)z * w * h * elempack + i * ((size_t)w * elempack) + j * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * w * h * out_elempack + j * ((size_t)h * out_elempack) + i * out_elempack;
                            permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, std::min(row_block, h - i), std::min(col_block, w - j), elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 2)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, d, h, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == out_elempack)
            {
                if (num_threads == 1)
                {
                    for (int q = 0; q < channels; q++)
                    {
                        const unsigned short* ptr = bottom_blob.channel(q);
                        unsigned short* outptr = top_blob.channel(q);
                        permute_transpose_blocks_bf16s_fp16s(ptr, (size_t)w * h * elempack, outptr, (size_t)w * d * elempack, d, h, w * elempack);
                    }
                    return 0;
                }
                if (channels >= num_threads)
                {
                    #pragma omp parallel for num_threads(num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const unsigned short* ptr = bottom_blob.channel(q);
                        unsigned short* outptr = top_blob.channel(q);
                        permute_transpose_blocks_bf16s_fp16s(ptr, (size_t)w * h * elempack, outptr, (size_t)w * d * elempack, d, h, w * elempack);
                    }
                }
                else if (channels == 1)
                {
                    permute_transpose_matrix_bf16s_fp16s(bottom_blob, (size_t)w * h * elempack, top_blob, (size_t)w * d * elempack, d, h, w * elempack, num_threads);
                }
                else
                {
                    const int col_block = permute_block_size(h, (size_t)d * (w * elempack) * sizeof(unsigned short), channels, num_threads, 4);
                    const int row_block = permute_block_size(d, (size_t)col_block * (w * elempack) * sizeof(unsigned short), channels * ((h + col_block - 1) / col_block), num_threads, 64 / sizeof(unsigned short));
                    #pragma omp parallel for collapse(3) num_threads(num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        for (int j = 0; j < h; j += col_block)
                        {
                            for (int i = 0; i < d; i += row_block)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + i * ((size_t)w * h * elempack) + (size_t)j * (w * elempack);
                                unsigned short* outptr = (unsigned short*)top_blob.channel(q) + j * ((size_t)w * d * elempack) + (size_t)i * (w * elempack);
                                permute_transpose_blocks_bf16s_fp16s(ptr, (size_t)w * h * elempack, outptr, (size_t)w * d * elempack, std::min(row_block, d - i), std::min(col_block, h - j), w * elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, elemsize, channels * h * d, num_threads, 32);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int i = 0; i < w; i += block)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + ((size_t)z * h + y) * w * elempack + i * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (((size_t)y * d + z) * w + i) * out_elempack;
                            permute_unpack_spatial_bf16s_fp16s(ptr, outptr, top_blob.cstep, std::min(block, w - i), elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 3)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, w, h, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (num_threads == 1)
            {
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)y * w * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * w * d * out_elempack;
                        permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * h * elempack, outptr, (size_t)d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    }
                }
                return 0;
            }
            if (channels * h >= num_threads)
            {
                #pragma omp parallel for collapse(2) num_threads(num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)y * w * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * w * d * out_elempack;
                        permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * h * elempack, outptr, (size_t)d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    }
                }
                return 0;
            }

            const int row_block = permute_block_size(d, (size_t)w * elemsize, channels * h, num_threads, 32);
            const int col_block = permute_block_size(w, (size_t)d * elemsize, channels * h * ((d + row_block - 1) / row_block), num_threads, 32);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int i = 0; i < d; i += row_block)
                    {
                        for (int j = 0; j < w; j += col_block)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)y * w * elempack + i * ((size_t)w * h * elempack) + j * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * w * d * out_elempack + j * ((size_t)d * out_elempack) + i * out_elempack;
                            permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * h * elempack, outptr, (size_t)d * out_elempack, top_blob.cstep, std::min(row_block, d - i), std::min(col_block, w - j), elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 4)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, d, w, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (num_threads == 1)
            {
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)z * w * h * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * h * out_elempack;
                        permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    }
                }
                return 0;
            }
            if (channels * d >= num_threads)
            {
                #pragma omp parallel for collapse(2) num_threads(num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)z * w * h * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * h * out_elempack;
                        permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    }
                }
                return 0;
            }

            const int row_block = permute_block_size(h, (size_t)w * elemsize, channels * d, num_threads, 32);
            const int col_block = permute_block_size(w, (size_t)h * elemsize, channels * d * ((h + row_block - 1) / row_block), num_threads, 32);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i += row_block)
                    {
                        for (int j = 0; j < w; j += col_block)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)z * w * h * elempack + i * ((size_t)w * elempack) + j * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)z * h * out_elempack + j * ((size_t)h * d * out_elempack) + i * out_elempack;
                            permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, std::min(row_block, h - i), std::min(col_block, w - j), elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 5)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, h, w, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (num_threads == 1)
            {
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)y * w * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * d * out_elempack;
                        permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * h * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    }
                }
                return 0;
            }
            if (channels * h >= num_threads)
            {
                #pragma omp parallel for collapse(2) num_threads(num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)y * w * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * d * out_elempack;
                        permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * h * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    }
                }
                return 0;
            }

            const int row_block = permute_block_size(d, (size_t)w * elemsize, channels * h, num_threads, 32);
            const int col_block = permute_block_size(w, (size_t)d * elemsize, channels * h * ((d + row_block - 1) / row_block), num_threads, 32);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int i = 0; i < d; i += row_block)
                    {
                        for (int j = 0; j < w; j += col_block)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)y * w * elempack + i * ((size_t)w * h * elempack) + j * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob.channel(q * elempack / out_elempack) + (size_t)y * d * out_elempack + j * ((size_t)h * d * out_elempack) + i * out_elempack;
                            permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * h * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, std::min(row_block, d - i), std::min(col_block, w - j), elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 6)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, h, channels * elempack, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                // w and h stay adjacent on both sides
                permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, top_blob.cstep, channels, d, w * h, num_threads);
                return 0;
            }

            // w and h stay adjacent on both sides
            const int block = permute_block_size(w * h, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)(w * h) * elempack * out_elemsize, top_blob.c * ((w * h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < w * h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + (size_t)c * elempack * ((size_t)w * h * out_elempack) + i * out_elempack;
                        permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w * h - i), std::min(channel_block, channels - c),
                                              elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)w * h * out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 7)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, w, channels * elempack, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)h * sizeof(unsigned short), (channels) * (d), num_threads, 32);
                const int row_block = permute_block_size(h, (size_t)w * sizeof(unsigned short), ((channels) * (d)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < h; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int z = 0; z < d; z++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h + (size_t)i * ((size_t)w) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w * top_blob.h + z * top_blob.cstep + (size_t)j * ((size_t)top_blob.w) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w, outptr, (size_t)top_blob.w, std::min(row_block, h - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (w) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((w) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < h; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                                unsigned short* outptr = (unsigned short*)top_blob + x * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * out_elempack;
                                // exchange c and d, keeping h as the inner spatial axis
                                permute3d_bf16s_fp16s(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                                      (size_t)w * elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((h) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int y = 0; y < h; y++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob + y * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * ((size_t)top_blob.w * out_elempack);
                            // exchange c and d, keeping w as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                                  elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 8)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, channels * elempack, h, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                if (num_threads == 1)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)z * w * h;
                        unsigned short* outptr = (unsigned short*)top_blob + z * top_blob.cstep;
                        permute_transpose_blocks_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, (size_t)channels * w, channels, h, w);
                    }
                    return 0;
                }
                if (d >= num_threads)
                {
                    #pragma omp parallel for num_threads(num_threads)
                    for (int z = 0; z < d; z++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)z * w * h;
                        unsigned short* outptr = (unsigned short*)top_blob + z * top_blob.cstep;
                        permute_transpose_blocks_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, (size_t)channels * w, channels, h, w);
                    }
                }
                else if (d == 1)
                {
                    permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, (size_t)channels * w, channels, h, w, num_threads);
                }
                else
                {
                    const int col_block = permute_block_size(h, (size_t)channels * (w) * sizeof(unsigned short), d, num_threads, 4);
                    const int row_block = permute_block_size(channels, (size_t)col_block * (w) * sizeof(unsigned short), d * ((h + col_block - 1) / col_block), num_threads, 64 / sizeof(unsigned short));
                    #pragma omp parallel for collapse(3) num_threads(num_threads)
                    for (int z = 0; z < d; z++)
                    {
                        for (int j = 0; j < h; j += col_block)
                        {
                            for (int i = 0; i < channels; i += row_block)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)z * w * h + i * (bottom_blob.cstep) + (size_t)j * (w);
                                unsigned short* outptr = (unsigned short*)top_blob + z * top_blob.cstep + j * ((size_t)channels * w) + (size_t)i * (w);
                                permute_transpose_blocks_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, (size_t)channels * w, std::min(row_block, channels - i), std::min(col_block, h - j), w);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((h) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int y = 0; y < h; y++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob + y * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                            // exchange c and d, keeping w as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                                  elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 9)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, w, h, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w * h, (size_t)channels * sizeof(unsigned short), d, num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)(w * h) * sizeof(unsigned short), d * ((w * h + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w * h; j += col_block)
                    {
                        for (int q = 0; q < d; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)q * w * h + (size_t)i * (bottom_blob.cstep) + j;
                            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + (size_t)j * (channels) + i;
                            permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, channels, std::min(row_block, channels - i), std::min(col_block, w * h - j));
                        }
                    }
                }
                return 0;
            }

            // w and h stay adjacent on both sides
            const int block = permute_block_size(w * h, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)(w * h) * elempack * out_elemsize, top_blob.c * ((w * h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < w * h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + (size_t)c * elempack * out_elempack + i * ((size_t)channels * elempack * out_elempack);
                        permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w * h - i), std::min(channel_block, channels - c),
                                              elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, (size_t)channels * elempack * out_elempack, out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 10)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, channels * elempack, w, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)h * sizeof(unsigned short), (channels) * (d), num_threads, 32);
                const int row_block = permute_block_size(h, (size_t)w * sizeof(unsigned short), ((channels) * (d)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < h; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int z = 0; z < d; z++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h + (size_t)i * ((size_t)w) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w + z * top_blob.cstep + (size_t)j * ((size_t)top_blob.w * top_blob.h) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w, outptr, (size_t)top_blob.w * top_blob.h, std::min(row_block, h - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (w) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((w) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < h; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                                unsigned short* outptr = (unsigned short*)top_blob + x * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                                // exchange c and d, keeping h as the inner spatial axis
                                permute3d_bf16s_fp16s(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                                      (size_t)w * elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((h) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int y = 0; y < h; y++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob + y * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and d, keeping w as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                                  elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 11)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = d % 16 == 0 ? 16 : d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = d % 8 == 0 ? 8 : d % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = d % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, h, w, d / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)channels * sizeof(unsigned short), (d) * (h), num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(unsigned short), ((d) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + z * top_blob.cstep + y * (size_t)top_blob.w + (size_t)j * ((size_t)top_blob.w * top_blob.h) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w * top_blob.h, std::min(row_block, channels - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((h) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int y = 0; y < h; y++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob + y * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and d, keeping w as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                                  elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 12)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, d, channels * elempack, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                // c and d have the same order on both sides; merge only without input padding
                if (bottom_blob.cstep == (size_t)w * h * d && (size_t)channels * d <= INT_MAX)
                {
                    permute_transpose_matrix_bf16s_fp16s(bottom_blob, (size_t)w * h, top_blob, top_blob.cstep, channels * d, h, w, num_threads);
                    return 0;
                }

                if (num_threads == 1)
                {
                    for (int q = 0; q < channels; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep;
                        unsigned short* outptr = (unsigned short*)top_blob + (size_t)q * d * w;
                        permute_transpose_blocks_bf16s_fp16s(ptr, (size_t)w * h, outptr, top_blob.cstep, d, h, w);
                    }
                    return 0;
                }
                if (channels >= num_threads)
                {
                    #pragma omp parallel for num_threads(num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep;
                        unsigned short* outptr = (unsigned short*)top_blob + (size_t)q * d * w;
                        permute_transpose_blocks_bf16s_fp16s(ptr, (size_t)w * h, outptr, top_blob.cstep, d, h, w);
                    }
                }
                else if (channels == 1)
                {
                    permute_transpose_matrix_bf16s_fp16s(bottom_blob, (size_t)w * h, top_blob, top_blob.cstep, d, h, w, num_threads);
                }
                else
                {
                    const int col_block = permute_block_size(h, (size_t)d * (w) * sizeof(unsigned short), channels, num_threads, 4);
                    const int row_block = permute_block_size(d, (size_t)col_block * (w) * sizeof(unsigned short), channels * ((h + col_block - 1) / col_block), num_threads, 64 / sizeof(unsigned short));
                    #pragma omp parallel for collapse(3) num_threads(num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        for (int j = 0; j < h; j += col_block)
                        {
                            for (int i = 0; i < d; i += row_block)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + i * ((size_t)w * h) + (size_t)j * (w);
                                unsigned short* outptr = (unsigned short*)top_blob + (size_t)q * d * w + j * (top_blob.cstep) + (size_t)i * (w);
                                permute_transpose_blocks_bf16s_fp16s(ptr, (size_t)w * h, outptr, top_blob.cstep, std::min(row_block, d - i), std::min(col_block, h - j), w);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * out_elempack;
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                                  elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 13)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, w, channels * elempack, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)d * sizeof(unsigned short), (channels) * (h), num_threads, 32);
                const int row_block = permute_block_size(d, (size_t)w * sizeof(unsigned short), ((channels) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w + (size_t)i * ((size_t)w * h) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w * top_blob.h + y * top_blob.cstep + (size_t)j * ((size_t)top_blob.w) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w * h, outptr, (size_t)top_blob.w, std::min(row_block, d - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize, (w) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize, ((w) * (top_blob.c)) * ((d + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * h * elempack);
                                unsigned short* outptr = (unsigned short*)top_blob + x * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * out_elempack;
                                // exchange c and h, keeping d as the inner spatial axis
                                permute3d_bf16s_fp16s(ptr, outptr, std::min(block, d - i), std::min(channel_block, channels - c),
                                                      (size_t)w * h * elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * ((size_t)top_blob.w * out_elempack);
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                                  elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 14)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, channels * elempack, d, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                if (num_threads == 1)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)z * w * h;
                        unsigned short* outptr = (unsigned short*)top_blob + (size_t)z * channels * w;
                        permute_transpose_blocks_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, top_blob.cstep, channels, h, w);
                    }
                    return 0;
                }
                if (d >= num_threads)
                {
                    #pragma omp parallel for num_threads(num_threads)
                    for (int z = 0; z < d; z++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)z * w * h;
                        unsigned short* outptr = (unsigned short*)top_blob + (size_t)z * channels * w;
                        permute_transpose_blocks_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, top_blob.cstep, channels, h, w);
                    }
                }
                else if (d == 1)
                {
                    permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, top_blob.cstep, channels, h, w, num_threads);
                }
                else
                {
                    const int col_block = permute_block_size(h, (size_t)channels * (w) * sizeof(unsigned short), d, num_threads, 4);
                    const int row_block = permute_block_size(channels, (size_t)col_block * (w) * sizeof(unsigned short), d * ((h + col_block - 1) / col_block), num_threads, 64 / sizeof(unsigned short));
                    #pragma omp parallel for collapse(3) num_threads(num_threads)
                    for (int z = 0; z < d; z++)
                    {
                        for (int j = 0; j < h; j += col_block)
                        {
                            for (int i = 0; i < channels; i += row_block)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)z * w * h + i * (bottom_blob.cstep) + (size_t)j * (w);
                                unsigned short* outptr = (unsigned short*)top_blob + (size_t)z * channels * w + j * (top_blob.cstep) + (size_t)i * (w);
                                permute_transpose_blocks_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, top_blob.cstep, std::min(row_block, channels - i), std::min(col_block, h - j), w);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                                  elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 15)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, w, d, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)channels * sizeof(unsigned short), (d) * (h), num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(unsigned short), ((d) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * top_blob.h + y * top_blob.cstep + (size_t)j * ((size_t)top_blob.w) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w, std::min(row_block, channels - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * out_elempack);
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                                  elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 16)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, channels * elempack, w, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)d * sizeof(unsigned short), (channels) * (h), num_threads, 32);
                const int row_block = permute_block_size(d, (size_t)w * sizeof(unsigned short), ((channels) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w + (size_t)i * ((size_t)w * h) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w + y * top_blob.cstep + (size_t)j * ((size_t)top_blob.w * top_blob.h) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w * h, outptr, (size_t)top_blob.w * top_blob.h, std::min(row_block, d - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize, (w) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize, ((w) * (top_blob.c)) * ((d + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * h * elempack);
                                unsigned short* outptr = (unsigned short*)top_blob + x * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                                // exchange c and h, keeping d as the inner spatial axis
                                permute3d_bf16s_fp16s(ptr, outptr, std::min(block, d - i), std::min(channel_block, channels - c),
                                                      (size_t)w * h * elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                                  elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 17)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = h % 16 == 0 ? 16 : h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = h % 8 == 0 ? 8 : h % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = h % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, d, w, h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)channels * sizeof(unsigned short), (d) * (h), num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(unsigned short), ((d) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w + y * top_blob.cstep + (size_t)j * ((size_t)top_blob.w * top_blob.h) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w * top_blob.h, std::min(row_block, channels - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(w, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)w * elempack * out_elemsize, ((d) * (top_blob.c)) * ((w + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < w; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * elempack;
                            unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and h, keeping w as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, w - i), std::min(channel_block, channels - c),
                                                  elempack, (size_t)w * elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 18)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, d, channels * elempack, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)(h * d) * sizeof(unsigned short), channels, num_threads, 32);
                const int row_block = permute_block_size(h * d, (size_t)w * sizeof(unsigned short), channels * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(3) num_threads(num_threads)
                for (int i = 0; i < h * d; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q) + (size_t)i * (w) + j;
                            unsigned short* outptr = (unsigned short*)top_blob + (size_t)q * h * d + (size_t)j * (top_blob.cstep) + i;
                            permute_transpose_pack1_bf16s_fp16s(ptr, w, outptr, top_blob.cstep, std::min(row_block, h * d - i), std::min(col_block, w - j));
                        }
                    }
                }
                return 0;
            }

            // h and d stay adjacent on both sides
            const int block = permute_block_size(h * d, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)(h * d) * elempack * out_elemsize, top_blob.c * ((h * d + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < h * d; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + (size_t)c * elempack * ((size_t)h * d * out_elempack) + i * out_elempack;
                        permute3d_bf16s_fp16s(ptr, outptr, std::min(block, h * d - i), std::min(channel_block, channels - c),
                                              (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)h * d * out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 19)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, h, channels * elempack, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)d * sizeof(unsigned short), (channels) * (h), num_threads, 32);
                const int row_block = permute_block_size(d, (size_t)w * sizeof(unsigned short), ((channels) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w + (size_t)i * ((size_t)w * h) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w * top_blob.h + y * (size_t)top_blob.w + (size_t)j * (top_blob.cstep) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w * h, outptr, top_blob.cstep, std::min(row_block, d - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize, ((h) * (top_blob.c)) * ((d + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int y = 0; y < h; y++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * h * elempack);
                                unsigned short* outptr = (unsigned short*)top_blob + y * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * out_elempack;
                                // exchange c and w, keeping d as the inner spatial axis
                                permute3d_bf16s_fp16s(ptr, outptr, std::min(block, d - i), std::min(channel_block, channels - c),
                                                      (size_t)w * h * elempack, elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((d) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                            unsigned short* outptr = (unsigned short*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * top_blob.h * out_elempack) + i * ((size_t)top_blob.w * out_elempack);
                            // exchange c and w, keeping h as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                                  (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 20)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, channels * elempack, d, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)h * sizeof(unsigned short), (channels) * (d), num_threads, 32);
                const int row_block = permute_block_size(h, (size_t)w * sizeof(unsigned short), ((channels) * (d)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < h; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int z = 0; z < d; z++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h + (size_t)i * ((size_t)w) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w + z * (size_t)top_blob.w * top_blob.h + (size_t)j * (top_blob.cstep) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w, outptr, top_blob.cstep, std::min(row_block, h - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((d) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                            unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                            // exchange c and w, keeping h as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                                  (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 21)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, h, d, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)channels * sizeof(unsigned short), (d) * (h), num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(unsigned short), ((d) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * top_blob.h + y * (size_t)top_blob.w + (size_t)j * (top_blob.cstep) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, top_blob.cstep, std::min(row_block, channels - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            // h and d stay adjacent on both sides
            const int block = permute_block_size(h * d, (size_t)channels * elempack * out_elemsize, top_blob.c, num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)(h * d) * elempack * out_elemsize, top_blob.c * ((h * d + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(3) num_threads(num_threads)
            for (int i = 0; i < h * d; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + (size_t)c * elempack * out_elempack + i * ((size_t)channels * elempack * out_elempack);
                        permute3d_bf16s_fp16s(ptr, outptr, std::min(block, h * d - i), std::min(channel_block, channels - c),
                                              (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, (size_t)channels * elempack * out_elempack, out_elempack, elempack, out_elempack);
                    }
                }
            }
            return 0;
        }

        if (order_type == 22)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, channels * elempack, h, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)d * sizeof(unsigned short), (channels) * (h), num_threads, 32);
                const int row_block = permute_block_size(d, (size_t)w * sizeof(unsigned short), ((channels) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int q = 0; q < channels; q++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w + (size_t)i * ((size_t)w * h) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w + y * (size_t)top_blob.w * top_blob.h + (size_t)j * (top_blob.cstep) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w * h, outptr, top_blob.cstep, std::min(row_block, d - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                const int block = permute_block_size(d, (size_t)channels * elempack * out_elemsize, (h) * (top_blob.c), num_threads, 32);
                const int channel_block = permute_block_size(channels, (size_t)d * elempack * out_elemsize, ((h) * (top_blob.c)) * ((d + block - 1) / block), num_threads, 16);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < d; i += block)
                {
                    for (int c = 0; c < channels; c += channel_block)
                    {
                        for (int y = 0; y < h; y++)
                        {
                            for (int q = 0; q < top_blob.c; q++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * h * elempack);
                                unsigned short* outptr = (unsigned short*)top_blob + y * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * out_elempack;
                                // exchange c and w, keeping d as the inner spatial axis
                                permute3d_bf16s_fp16s(ptr, outptr, std::min(block, d - i), std::min(channel_block, channels - c),
                                                      (size_t)w * h * elempack, elempack, bottom_blob.cstep * elempack, out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((d) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                            unsigned short* outptr = (unsigned short*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * ((size_t)top_blob.w * out_elempack) + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and w, keeping h as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                                  (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }

        if (order_type == 23)
        {
            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
#if __AVX512F__
                out_elempack = w % 16 == 0 ? 16 : w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __AVX__
                out_elempack = w % 8 == 0 ? 8 : w % 4 == 0 ? 4 : 1;
#elif __SSE2__
                out_elempack = w % 4 == 0 ? 4 : 1;
#endif
            }
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(channels * elempack, d, h, w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (elempack == 1 && out_elempack == 1)
            {
                const int col_block = permute_block_size(w, (size_t)channels * sizeof(unsigned short), (d) * (h), num_threads, 32);
                const int row_block = permute_block_size(channels, (size_t)w * sizeof(unsigned short), ((d) * (h)) * ((w + col_block - 1) / col_block), num_threads, 32);
                #pragma omp parallel for collapse(4) num_threads(num_threads)
                for (int i = 0; i < channels; i += row_block)
                {
                    for (int j = 0; j < w; j += col_block)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            for (int y = 0; y < h; y++)
                            {
                                const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w + (size_t)i * (bottom_blob.cstep) + j;
                                unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w + y * (size_t)top_blob.w * top_blob.h + (size_t)j * (top_blob.cstep) + i;
                                permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, top_blob.cstep, std::min(row_block, channels - i), std::min(col_block, w - j));
                            }
                        }
                    }
                }
                return 0;
            }

            const int block = permute_block_size(h, (size_t)channels * elempack * out_elemsize, (d) * (top_blob.c), num_threads, 32);
            const int channel_block = permute_block_size(channels, (size_t)h * elempack * out_elemsize, ((d) * (top_blob.c)) * ((h + block - 1) / block), num_threads, 16);
            #pragma omp parallel for collapse(4) num_threads(num_threads)
            for (int i = 0; i < h; i += block)
            {
                for (int c = 0; c < channels; c += channel_block)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int q = 0; q < top_blob.c; q++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack + (size_t)c * (bottom_blob.cstep * elempack) + i * ((size_t)w * elempack);
                            unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack + (size_t)c * elempack * out_elempack + i * ((size_t)top_blob.w * top_blob.h * out_elempack);
                            // exchange c and w, keeping h as the inner spatial axis
                            permute3d_bf16s_fp16s(ptr, outptr, std::min(block, h - i), std::min(channel_block, channels - c),
                                                  (size_t)w * elempack, elempack, bottom_blob.cstep * elempack, (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
                        }
                    }
                }
            }
            return 0;
        }
    }

    return -1;
}

} // namespace ncnn
