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
    const int nT = bottom_blob.total() * elemsize >= 65536 ? opt.num_threads : 1;
#else
    const int nT = 1;
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

            permute2d(bottom_blob, top_blob, nT);
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

            permute3d_hwc(bottom_blob, top_blob, nT);
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

            permute3d_wch(bottom_blob, top_blob, nT);
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

            permute3d_cwh(bottom_blob, top_blob, nT);
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

            permute3d_hcw(bottom_blob, top_blob, nT);
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

            permute3d_chw(bottom_blob, top_blob, nT);
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

            permute4d_hwdc(bottom_blob, top_blob, nT);
            return 0;
        }

        if (order_type == 2)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, d, h, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            permute4d_wdhc(bottom_blob, top_blob, nT);
            return 0;
        }

        if (order_type == 3)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, w, h, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            permute4d_dwhc(bottom_blob, top_blob, nT);
            return 0;
        }

        if (order_type == 4)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, d, w, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            permute4d_hdwc(bottom_blob, top_blob, nT);
            return 0;
        }

        if (order_type == 5)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, h, w, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            permute4d_dhwc(bottom_blob, top_blob, nT);
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

            permute4d_whcd(bottom_blob, top_blob, nT);
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

            permute4d_hwcd(bottom_blob, top_blob, nT);
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

            permute4d_wchd(bottom_blob, top_blob, nT);
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

            permute4d_cwhd(bottom_blob, top_blob, nT);
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

            permute4d_hcwd(bottom_blob, top_blob, nT);
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

            permute4d_chwd(bottom_blob, top_blob, nT);
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

            permute4d_wdch(bottom_blob, top_blob, nT);
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

            permute4d_dwch(bottom_blob, top_blob, nT);
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

            permute4d_wcdh(bottom_blob, top_blob, nT);
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

            permute4d_cwdh(bottom_blob, top_blob, nT);
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

            permute4d_dcwh(bottom_blob, top_blob, nT);
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

            permute4d_cdwh(bottom_blob, top_blob, nT);
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

            permute4d_hdcw(bottom_blob, top_blob, nT);
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

            permute4d_dhcw(bottom_blob, top_blob, nT);
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

            permute4d_hcdw(bottom_blob, top_blob, nT);
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

            permute4d_chdw(bottom_blob, top_blob, nT);
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

            permute4d_dchw(bottom_blob, top_blob, nT);
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

            permute4d_cdhw(bottom_blob, top_blob, nT);
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
    const int nT = bottom_blob.total() * elemsize >= 65536 ? opt.num_threads : 1;
#else
    const int nT = 1;
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

            permute2d_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute3d_hwc_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute3d_wch_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute3d_cwh_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute3d_hcw_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute3d_chw_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_hwdc_bf16s_fp16s(bottom_blob, top_blob, nT);
            return 0;
        }

        if (order_type == 2)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(w, d, h, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            permute4d_wdhc_bf16s_fp16s(bottom_blob, top_blob, nT);
            return 0;
        }

        if (order_type == 3)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, w, h, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            permute4d_dwhc_bf16s_fp16s(bottom_blob, top_blob, nT);
            return 0;
        }

        if (order_type == 4)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(h, d, w, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            permute4d_hdwc_bf16s_fp16s(bottom_blob, top_blob, nT);
            return 0;
        }

        if (order_type == 5)
        {
            const int out_elempack = opt.use_packing_layout ? elempack : 1;
            const size_t out_elemsize = elemsize / elempack * out_elempack;
            top_blob.create(d, h, w, channels * elempack / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            permute4d_dhwc_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_whcd_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_hwcd_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_wchd_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_cwhd_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_hcwd_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_chwd_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_wdch_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_dwch_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_wcdh_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_cwdh_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_dcwh_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_cdwh_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_hdcw_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_dhcw_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_hcdw_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_chdw_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_dchw_bf16s_fp16s(bottom_blob, top_blob, nT);
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

            permute4d_cdhw_bf16s_fp16s(bottom_blob, top_blob, nT);
            return 0;
        }
    }

    return -1;
}

} // namespace ncnn
