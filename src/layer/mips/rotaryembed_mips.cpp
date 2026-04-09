// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_mips.h"

namespace ncnn {

RotaryEmbed_mips::RotaryEmbed_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static int unpack_or_cast_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    if (src.empty())
    {
        dst = src;
        return 0;
    }

    Mat unpacked = src;
    if (src.elempack != 1)
    {
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;

        convert_packing(src, unpacked, 1, opt_unpack);
        if (unpacked.empty())
            return -100;
    }

#if NCNN_BF16
    if (unpacked.elembits() == 16)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = opt.workspace_allocator;

        cast_bfloat16_to_float32(unpacked, dst, opt_cast);
        if (dst.empty())
            return -100;
        return 0;
    }
#endif

    dst = unpacked;
    return 0;
}

int RotaryEmbed_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    Mat bottom_blob_fp32;
    Mat cos_cache_fp32;
    Mat sin_cache_fp32;

    if (unpack_or_cast_to_float32(bottom_blob, bottom_blob_fp32, opt) != 0)
        return -100;
    if (unpack_or_cast_to_float32(cos_cache, cos_cache_fp32, opt) != 0)
        return -100;
    if (unpack_or_cast_to_float32(sin_cache, sin_cache_fp32, opt) != 0)
        return -100;

    const int need_postprocess = bottom_blob.elempack != 1 || (opt.use_bf16_storage && bottom_blob.elembits() == 16);

    Option opt_fp32 = opt;
    if (need_postprocess)
        opt_fp32.blob_allocator = opt.workspace_allocator;

    const int embed_dim = bottom_blob_fp32.w;
    const int seqlen = bottom_blob_fp32.h;
    const int num_heads = bottom_blob_fp32.c;

    Mat top_blob_fp32;
    top_blob_fp32.create_like(bottom_blob_fp32, opt_fp32.blob_allocator);
    if (top_blob_fp32.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob_fp32.channel(q);
        Mat out_head = top_blob_fp32.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            if (interleaved)
            {
                const float* ptr = head.row(i);
                const float* cos_ptr = cos_cache_fp32.row(i);
                const float* sin_ptr = sin_cache_fp32.row(i);
                float* outptr = out_head.row(i);

                for (int j = 0; j < embed_dim / 2; j++)
                {
                    const float x0 = ptr[0];
                    const float x1 = ptr[1];
                    const float cos_val = *cos_ptr++;
                    const float sin_val = *sin_ptr++;
                    outptr[0] = x0 * cos_val - x1 * sin_val;
                    outptr[1] = x0 * sin_val + x1 * cos_val;
                    ptr += 2;
                    outptr += 2;
                }
            }
            else
            {
                const float* ptr0 = head.row(i);
                const float* ptr1 = ptr0 + embed_dim / 2;
                const float* cos_ptr = cos_cache_fp32.row(i);
                const float* sin_ptr = sin_cache_fp32.row(i);
                float* outptr0 = out_head.row(i);
                float* outptr1 = outptr0 + embed_dim / 2;

                for (int j = 0; j < embed_dim / 2; j++)
                {
                    const float x0 = *ptr0++;
                    const float x1 = *ptr1++;
                    const float cos_val = *cos_ptr++;
                    const float sin_val = *sin_ptr++;
                    *outptr0++ = x0 * cos_val - x1 * sin_val;
                    *outptr1++ = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    Mat top_blob_packed = top_blob_fp32;
    if (bottom_blob.elempack != 1)
    {
        Option opt_pack = opt;
        if (opt.use_bf16_storage && bottom_blob.elembits() == 16)
            opt_pack.blob_allocator = opt.workspace_allocator;

        convert_packing(top_blob_fp32, top_blob_packed, bottom_blob.elempack, opt_pack);
        if (top_blob_packed.empty())
            return -100;
    }

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blob.elembits() == 16)
    {
        cast_float32_to_bfloat16(top_blob_packed, top_blobs[0], opt);
        if (top_blobs[0].empty())
            return -100;
        return 0;
    }
#endif

    top_blobs[0] = top_blob_packed;
    return 0;
}

} // namespace ncnn
