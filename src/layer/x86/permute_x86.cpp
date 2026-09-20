// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "permute_x86.h"

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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int x = 0; x < w; x += 32)
            {
                const float* ptr = (const float*)bottom_blob + x * elempack;
                float* outptr = top_blob.row<float>(x / out_elempack);
                permute_transpose2d(ptr, (size_t)w * elempack, outptr, (size_t)top_blob.w * out_elempack, h * elempack, std::min(32, w - x), elempack, out_elempack);
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q * elempack / out_elempack);
                permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    float* outptr = top_blob.channel(q);
                    for (int y = 0; y < top_blob.h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * (size_t)w + y * bottom_blob.cstep;
                        memcpy(outptr, ptr, (size_t)w * sizeof(float));
                        outptr += w;
                    }
                }
                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + q * out_elempack * (size_t)w * elempack;
                float* outptr = top_blob.channel(q);
                // Exchange c and h, keeping w as the inner spatial axis.
                permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = (const float*)bottom_blob + y * (size_t)w;
                    float* outptr = (float*)top_blob + y * top_blob.cstep;
                    permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w, channels, w);
                }
                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + q * out_elempack * (size_t)w * elempack;
                float* outptr = top_blob.channel(q);
                // Exchange c and h, keeping w as the inner spatial axis.
                permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep;
                    float* outptr = (float*)top_blob + q * (size_t)top_blob.w;
                    permute_transpose_pack1(ptr, (size_t)w, outptr, top_blob.cstep, h, w);
                }
                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + q * out_elempack * elempack;
                float* outptr = top_blob.channel(q);
                // Exchange c and w, keeping h as the inner spatial axis.
                permute3d(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = (const float*)bottom_blob + y * (size_t)w;
                    float* outptr = (float*)top_blob + y * (size_t)top_blob.w;
                    permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, top_blob.cstep, channels, w);
                }
                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + q * out_elempack * elempack;
                float* outptr = top_blob.channel(q);
                // Exchange c and w, keeping h as the inner spatial axis.
                permute3d(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q * elempack / out_elempack);
                for (int z = 0; z < d; z++)
                {
                    permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    ptr += (size_t)w * h * elempack;
                    outptr += (size_t)w * h * out_elempack;
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q * elempack / out_elempack);
                for (int y = 0; y < h; y++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        permute_copy_spatial(ptr + ((size_t)z * h + y) * w * elempack, outptr, top_blob.cstep, w, elempack, out_elempack);
                        outptr += w * out_elempack;
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q * elempack / out_elempack);
                for (int y = 0; y < h; y++)
                {
                    permute_transpose_spatial(ptr, (size_t)w * h * elempack, outptr, (size_t)d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    ptr += (size_t)w * elempack;
                    outptr += (size_t)w * d * out_elempack;
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q * elempack / out_elempack);
                for (int z = 0; z < d; z++)
                {
                    permute_transpose_spatial(ptr, (size_t)w * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    ptr += (size_t)w * h * elempack;
                    outptr += (size_t)h * out_elempack;
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q * elempack / out_elempack);
                for (int y = 0; y < h; y++)
                {
                    permute_transpose_spatial(ptr, (size_t)w * h * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    ptr += (size_t)w * elempack;
                    outptr += (size_t)d * out_elempack;
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < d; q++)
                {
                    float* outptr = top_blob.channel(q);
                    for (int c = 0; c < channels; c++)
                    {
                        const float* ptr = bottom_blob.channel(c).depth(q);
                        memcpy(outptr, ptr, (size_t)w * h * sizeof(float));
                        outptr += (size_t)w * h;
                    }
                }
                return 0;
            }

// w and h stay adjacent on both sides.
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + q * out_elempack * (size_t)w * h * elempack;
                float* outptr = top_blob.channel(q);
                permute3d(ptr, outptr, w * h, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)w * h * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h;
                        float* outptr = (float*)top_blob + q * (size_t)top_blob.w * top_blob.h + z * top_blob.cstep;
                        permute_transpose_pack1(ptr, (size_t)w, outptr, (size_t)top_blob.w, h, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * h * elempack;
                        float* outptr = (float*)top_blob + x * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and d, keeping h as the inner spatial axis.
                        permute3d(ptr, outptr, h, channels,
                               (size_t)w * elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack;
                    float* outptr = (float*)top_blob + y * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and d, keeping w as the inner spatial axis.
                    permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    float* outptr = top_blob.channel(q);
                    for (int z = 0; z < top_blob.d; z++)
                    {
                        for (int y = 0; y < top_blob.h; y++)
                        {
                            const float* ptr = (const float*)bottom_blob + q * (size_t)w * h + z * (size_t)w + y * bottom_blob.cstep;
                            memcpy(outptr, ptr, (size_t)w * sizeof(float));
                            outptr += w;
                        }
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack;
                    float* outptr = (float*)top_blob + y * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and d, keeping w as the inner spatial axis.
                    permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < d; q++)
                {
                    const float* ptr = (const float*)bottom_blob + (size_t)q * w * h;
                    float* outptr = top_blob.channel(q);
                    permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, channels, channels, w * h);
                }
                return 0;
            }

// w and h stay adjacent on both sides.
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + q * out_elempack * (size_t)w * h * elempack;
                float* outptr = top_blob.channel(q);
                permute3d(ptr, outptr, w * h, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               (size_t)channels * elempack * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h;
                        float* outptr = (float*)top_blob + q * (size_t)top_blob.w + z * top_blob.cstep;
                        permute_transpose_pack1(ptr, (size_t)w, outptr, (size_t)top_blob.w * top_blob.h, h, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * h * elempack;
                        float* outptr = (float*)top_blob + x * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and d, keeping h as the inner spatial axis.
                        permute3d(ptr, outptr, h, channels,
                               (size_t)w * elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack;
                    float* outptr = (float*)top_blob + y * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and d, keeping w as the inner spatial axis.
                    permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob + z * (size_t)w * h + y * (size_t)w;
                        float* outptr = (float*)top_blob + z * top_blob.cstep + y * (size_t)top_blob.w;
                        permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w * top_blob.h, channels, w);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack;
                    float* outptr = (float*)top_blob + y * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and d, keeping w as the inner spatial axis.
                    permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    float* outptr = top_blob.channel(q);
                    for (int z = 0; z < top_blob.d; z++)
                    {
                        for (int y = 0; y < top_blob.h; y++)
                        {
                            const float* ptr = (const float*)bottom_blob + q * (size_t)w + z * bottom_blob.cstep + y * (size_t)w * h;
                            memcpy(outptr, ptr, (size_t)w * sizeof(float));
                            outptr += w;
                        }
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    float* outptr = (float*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w;
                        float* outptr = (float*)top_blob + q * (size_t)top_blob.w * top_blob.h + y * top_blob.cstep;
                        permute_transpose_pack1(ptr, (size_t)w * h, outptr, (size_t)top_blob.w, d, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * elempack;
                        float* outptr = (float*)top_blob + x * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and h, keeping d as the inner spatial axis.
                        permute3d(ptr, outptr, d, channels,
                               (size_t)w * h * elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    float* outptr = (float*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    float* outptr = top_blob.channel(q);
                    for (int z = 0; z < top_blob.d; z++)
                    {
                        for (int y = 0; y < top_blob.h; y++)
                        {
                            const float* ptr = (const float*)bottom_blob + q * (size_t)w + z * (size_t)w * h + y * bottom_blob.cstep;
                            memcpy(outptr, ptr, (size_t)w * sizeof(float));
                            outptr += w;
                        }
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    float* outptr = (float*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob + z * (size_t)w * h + y * (size_t)w;
                        float* outptr = (float*)top_blob + z * (size_t)top_blob.w * top_blob.h + y * top_blob.cstep;
                        permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w, channels, w);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    float* outptr = (float*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w;
                        float* outptr = (float*)top_blob + q * (size_t)top_blob.w + y * top_blob.cstep;
                        permute_transpose_pack1(ptr, (size_t)w * h, outptr, (size_t)top_blob.w * top_blob.h, d, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * elempack;
                        float* outptr = (float*)top_blob + x * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and h, keeping d as the inner spatial axis.
                        permute3d(ptr, outptr, d, channels,
                               (size_t)w * h * elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    float* outptr = (float*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob + z * (size_t)w * h + y * (size_t)w;
                        float* outptr = (float*)top_blob + z * (size_t)top_blob.w + y * top_blob.cstep;
                        permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w * top_blob.h, channels, w);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    float* outptr = (float*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);
                    float* outptr = (float*)top_blob + (size_t)q * h * d;
                    permute_transpose_pack1(ptr, w, outptr, top_blob.cstep, h * d, w);
                }
                return 0;
            }

// h and d stay adjacent on both sides.
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + q * out_elempack * elempack;
                float* outptr = top_blob.channel(q);
                permute3d(ptr, outptr, h * d, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)h * d * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w;
                        float* outptr = (float*)top_blob + q * (size_t)top_blob.w * top_blob.h + y * (size_t)top_blob.w;
                        permute_transpose_pack1(ptr, (size_t)w * h, outptr, top_blob.cstep, d, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * elempack;
                        float* outptr = (float*)top_blob + y * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and w, keeping d as the inner spatial axis.
                        permute3d(ptr, outptr, d, channels,
                               (size_t)w * h * elempack, elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack;
                    float* outptr = (float*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and w, keeping h as the inner spatial axis.
                    permute3d(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h;
                        float* outptr = (float*)top_blob + q * (size_t)top_blob.w + z * (size_t)top_blob.w * top_blob.h;
                        permute_transpose_pack1(ptr, (size_t)w, outptr, top_blob.cstep, h, w);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack;
                    float* outptr = (float*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and w, keeping h as the inner spatial axis.
                    permute3d(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob + z * (size_t)w * h + y * (size_t)w;
                        float* outptr = (float*)top_blob + z * (size_t)top_blob.w * top_blob.h + y * (size_t)top_blob.w;
                        permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, top_blob.cstep, channels, w);
                    }
                }
                return 0;
            }

// h and d stay adjacent on both sides.
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob + q * out_elempack * elempack;
                float* outptr = top_blob.channel(q);
                permute3d(ptr, outptr, h * d, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               (size_t)channels * elempack * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w;
                        float* outptr = (float*)top_blob + q * (size_t)top_blob.w + y * (size_t)top_blob.w * top_blob.h;
                        permute_transpose_pack1(ptr, (size_t)w * h, outptr, top_blob.cstep, d, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const float* ptr = (const float*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * elempack;
                        float* outptr = (float*)top_blob + y * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and w, keeping d as the inner spatial axis.
                        permute3d(ptr, outptr, d, channels,
                               (size_t)w * h * elempack, elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack;
                    float* outptr = (float*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and w, keeping h as the inner spatial axis.
                    permute3d(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float* ptr = (const float*)bottom_blob + z * (size_t)w * h + y * (size_t)w;
                        float* outptr = (float*)top_blob + z * (size_t)top_blob.w + y * (size_t)top_blob.w * top_blob.h;
                        permute_transpose_pack1(ptr, bottom_blob.cstep, outptr, top_blob.cstep, channels, w);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack;
                    float* outptr = (float*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and w, keeping h as the inner spatial axis.
                    permute3d(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int x = 0; x < w; x += 32)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + x * elempack;
                unsigned short* outptr = top_blob.row<unsigned short>(x / out_elempack);
                permute_transpose2d_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)top_blob.w * out_elempack, h * elempack, std::min(32, w - x), elempack, out_elempack);
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr = top_blob.channel(q * elempack / out_elempack);
                permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    unsigned short* outptr = top_blob.channel(q);
                    for (int y = 0; y < top_blob.h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * (size_t)w + y * bottom_blob.cstep;
                        memcpy(outptr, ptr, (size_t)w * sizeof(unsigned short));
                        outptr += w;
                    }
                }
                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * (size_t)w * elempack;
                unsigned short* outptr = top_blob.channel(q);
                // Exchange c and h, keeping w as the inner spatial axis.
                permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w;
                    unsigned short* outptr = (unsigned short*)top_blob + y * top_blob.cstep;
                    permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w, channels, w);
                }
                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * (size_t)w * elempack;
                unsigned short* outptr = top_blob.channel(q);
                // Exchange c and h, keeping w as the inner spatial axis.
                permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep;
                    unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w;
                    permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w, outptr, top_blob.cstep, h, w);
                }
                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * elempack;
                unsigned short* outptr = top_blob.channel(q);
                // Exchange c and w, keeping h as the inner spatial axis.
                permute3d_bf16s_fp16s(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w;
                    unsigned short* outptr = (unsigned short*)top_blob + y * (size_t)top_blob.w;
                    permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, top_blob.cstep, channels, w);
                }
                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * elempack;
                unsigned short* outptr = top_blob.channel(q);
                // Exchange c and w, keeping h as the inner spatial axis.
                permute3d_bf16s_fp16s(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr = top_blob.channel(q * elempack / out_elempack);
                for (int z = 0; z < d; z++)
                {
                    permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    ptr += (size_t)w * h * elempack;
                    outptr += (size_t)w * h * out_elempack;
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr = top_blob.channel(q * elempack / out_elempack);
                for (int y = 0; y < h; y++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        permute_copy_spatial_bf16s_fp16s(ptr + ((size_t)z * h + y) * w * elempack, outptr, top_blob.cstep, w, elempack, out_elempack);
                        outptr += w * out_elempack;
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr = top_blob.channel(q * elempack / out_elempack);
                for (int y = 0; y < h; y++)
                {
                    permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * h * elempack, outptr, (size_t)d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    ptr += (size_t)w * elempack;
                    outptr += (size_t)w * d * out_elempack;
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr = top_blob.channel(q * elempack / out_elempack);
                for (int z = 0; z < d; z++)
                {
                    permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, h, w, elempack, out_elempack);
                    ptr += (size_t)w * h * elempack;
                    outptr += (size_t)h * out_elempack;
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

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr = top_blob.channel(q * elempack / out_elempack);
                for (int y = 0; y < h; y++)
                {
                    permute_transpose_spatial_bf16s_fp16s(ptr, (size_t)w * h * elempack, outptr, (size_t)h * d * out_elempack, top_blob.cstep, d, w, elempack, out_elempack);
                    ptr += (size_t)w * elempack;
                    outptr += (size_t)d * out_elempack;
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < d; q++)
                {
                    unsigned short* outptr = top_blob.channel(q);
                    for (int c = 0; c < channels; c++)
                    {
                        const unsigned short* ptr = bottom_blob.channel(c).depth(q);
                        memcpy(outptr, ptr, (size_t)w * h * sizeof(unsigned short));
                        outptr += (size_t)w * h;
                    }
                }
                return 0;
            }

// w and h stay adjacent on both sides.
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * (size_t)w * h * elempack;
                unsigned short* outptr = top_blob.channel(q);
                permute3d_bf16s_fp16s(ptr, outptr, w * h, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)w * h * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h;
                        unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w * top_blob.h + z * top_blob.cstep;
                        permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w, outptr, (size_t)top_blob.w, h, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * h * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob + x * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and d, keeping h as the inner spatial axis.
                        permute3d_bf16s_fp16s(ptr, outptr, h, channels,
                               (size_t)w * elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + y * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and d, keeping w as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    unsigned short* outptr = top_blob.channel(q);
                    for (int z = 0; z < top_blob.d; z++)
                    {
                        for (int y = 0; y < top_blob.h; y++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + q * (size_t)w * h + z * (size_t)w + y * bottom_blob.cstep;
                            memcpy(outptr, ptr, (size_t)w * sizeof(unsigned short));
                            outptr += w;
                        }
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + y * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and d, keeping w as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < d; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)q * w * h;
                    unsigned short* outptr = top_blob.channel(q);
                    permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, channels, channels, w * h);
                }
                return 0;
            }

// w and h stay adjacent on both sides.
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * (size_t)w * h * elempack;
                unsigned short* outptr = top_blob.channel(q);
                permute3d_bf16s_fp16s(ptr, outptr, w * h, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               (size_t)channels * elempack * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h;
                        unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w + z * top_blob.cstep;
                        permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w, outptr, (size_t)top_blob.w * top_blob.h, h, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * h * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob + x * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and d, keeping h as the inner spatial axis.
                        permute3d_bf16s_fp16s(ptr, outptr, h, channels,
                               (size_t)w * elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + y * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and d, keeping w as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w;
                        unsigned short* outptr = (unsigned short*)top_blob + z * top_blob.cstep + y * (size_t)top_blob.w;
                        permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w * top_blob.h, channels, w);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * (size_t)w * h * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + y * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and d, keeping w as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * h * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    unsigned short* outptr = top_blob.channel(q);
                    for (int z = 0; z < top_blob.d; z++)
                    {
                        for (int y = 0; y < top_blob.h; y++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + q * (size_t)w + z * bottom_blob.cstep + y * (size_t)w * h;
                            memcpy(outptr, ptr, (size_t)w * sizeof(unsigned short));
                            outptr += w;
                        }
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w;
                        unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w * top_blob.h + y * top_blob.cstep;
                        permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w * h, outptr, (size_t)top_blob.w, d, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob + x * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and h, keeping d as the inner spatial axis.
                        permute3d_bf16s_fp16s(ptr, outptr, d, channels,
                               (size_t)w * h * elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    unsigned short* outptr = top_blob.channel(q);
                    for (int z = 0; z < top_blob.d; z++)
                    {
                        for (int y = 0; y < top_blob.h; y++)
                        {
                            const unsigned short* ptr = (const unsigned short*)bottom_blob + q * (size_t)w + z * (size_t)w * h + y * bottom_blob.cstep;
                            memcpy(outptr, ptr, (size_t)w * sizeof(unsigned short));
                            outptr += w;
                        }
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w;
                        unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * top_blob.h + y * top_blob.cstep;
                        permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w, channels, w);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w;
                        unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w + y * top_blob.cstep;
                        permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w * h, outptr, (size_t)top_blob.w * top_blob.h, d, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int x = 0; x < w; x++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + x * elempack + q * out_elempack * (size_t)w * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob + x * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and h, keeping d as the inner spatial axis.
                        permute3d_bf16s_fp16s(ptr, outptr, d, channels,
                               (size_t)w * h * elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w;
                        unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w + y * top_blob.cstep;
                        permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, (size_t)top_blob.w * top_blob.h, channels, w);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * (size_t)w * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and h, keeping w as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, w, channels,
                               elempack, (size_t)w * elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);
                    unsigned short* outptr = (unsigned short*)top_blob + (size_t)q * h * d;
                    permute_transpose_pack1_bf16s_fp16s(ptr, w, outptr, top_blob.cstep, h * d, w);
                }
                return 0;
            }

// h and d stay adjacent on both sides.
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * elempack;
                unsigned short* outptr = top_blob.channel(q);
                permute3d_bf16s_fp16s(ptr, outptr, h * d, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)h * d * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w;
                        unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w * top_blob.h + y * (size_t)top_blob.w;
                        permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w * h, outptr, top_blob.cstep, d, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob + y * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and w, keeping d as the inner spatial axis.
                        permute3d_bf16s_fp16s(ptr, outptr, d, channels,
                               (size_t)w * h * elempack, elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and w, keeping h as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * out_elempack, (size_t)top_blob.w * top_blob.h * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + z * (size_t)w * h;
                        unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w + z * (size_t)top_blob.w * top_blob.h;
                        permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w, outptr, top_blob.cstep, h, w);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and w, keeping h as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w;
                        unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * top_blob.h + y * (size_t)top_blob.w;
                        permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, top_blob.cstep, channels, w);
                    }
                }
                return 0;
            }

// h and d stay adjacent on both sides.
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + q * out_elempack * elempack;
                unsigned short* outptr = top_blob.channel(q);
                permute3d_bf16s_fp16s(ptr, outptr, h * d, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               (size_t)channels * elempack * out_elempack, out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + q * bottom_blob.cstep + y * (size_t)w;
                        unsigned short* outptr = (unsigned short*)top_blob + q * (size_t)top_blob.w + y * (size_t)top_blob.w * top_blob.h;
                        permute_transpose_pack1_bf16s_fp16s(ptr, (size_t)w * h, outptr, top_blob.cstep, d, w);
                    }
                }
                return 0;
            }

            if (out_elempack == 1)
            {
                #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    for (int q = 0; q < top_blob.c; q++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + y * (size_t)w * elempack + q * out_elempack * elempack;
                        unsigned short* outptr = (unsigned short*)top_blob + y * (size_t)top_blob.w * top_blob.h * out_elempack + q * top_blob.cstep * out_elempack;
                        // Exchange c and w, keeping d as the inner spatial axis.
                        permute3d_bf16s_fp16s(ptr, outptr, d, channels,
                               (size_t)w * h * elempack, elempack, bottom_blob.cstep * elempack,
                               out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + z * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and w, keeping h as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, (size_t)top_blob.w * out_elempack, elempack, out_elempack);
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h + y * (size_t)w;
                        unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w + y * (size_t)top_blob.w * top_blob.h;
                        permute_transpose_pack1_bf16s_fp16s(ptr, bottom_blob.cstep, outptr, top_blob.cstep, channels, w);
                    }
                }
                return 0;
            }

            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
            for (int z = 0; z < d; z++)
            {
                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h * elempack + q * out_elempack * elempack;
                    unsigned short* outptr = (unsigned short*)top_blob + z * (size_t)top_blob.w * out_elempack + q * top_blob.cstep * out_elempack;
                    // Exchange c and w, keeping h as the inner spatial axis.
                    permute3d_bf16s_fp16s(ptr, outptr, h, channels,
                               (size_t)w * elempack, elempack, bottom_blob.cstep * elempack,
                               (size_t)top_blob.w * top_blob.h * out_elempack, out_elempack, elempack, out_elempack);
                }
            }
            return 0;
        }
    }

    return -1;
}

} // namespace ncnn
