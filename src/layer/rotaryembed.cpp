// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed.h"

namespace ncnn {

RotaryEmbed::RotaryEmbed()
{
}

int RotaryEmbed::load_param(const ParamDict& pd)
{
    interleaved = pd.get(0, 0);
    input_hc_swapped = pd.get(1, 0);

    return 0;
}

int RotaryEmbed::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // assert bottom_blobs.size() == 3

    const Mat& bottom_blob = bottom_blobs[0];

    /* The fused (input_hc_swapped) path below is fp32/pack1 only. Backends that
     * do not implement it delegate here, so a 16-bit or packed fused input must
     * not fall into the fp32 loops: that would read the storage as float and
     * write twice the row width into a 16-bit output blob. Convert to
     * fp32/pack1, run the same code, and convert the result back. */
    if (input_hc_swapped && (bottom_blob.elembits() != 32 || bottom_blob.elempack != 1))
    {
        Mat q32, cos32, sin32, top32;

        const Mat& cos_cache = bottom_blobs[1];
        const Mat& sin_cache = bottom_blobs[2];

        if (opt.use_bf16_storage && bottom_blob.elembits() == 16)
            cast_bfloat16_to_float32(bottom_blob, q32, opt);
        else
            cast_float16_to_float32(bottom_blob, q32, opt);
        if (cos_cache.elembits() != 32)
        {
            if (opt.use_bf16_storage && cos_cache.elembits() == 16)
                cast_bfloat16_to_float32(cos_cache, cos32, opt);
            else
                cast_float16_to_float32(cos_cache, cos32, opt);
        }
        else
            cos32 = cos_cache;
        if (sin_cache.elembits() != 32)
        {
            if (opt.use_bf16_storage && sin_cache.elembits() == 16)
                cast_bfloat16_to_float32(sin_cache, sin32, opt);
            else
                cast_float16_to_float32(sin_cache, sin32, opt);
        }
        else
            sin32 = sin_cache;

        /* cast keeps the packing; the fused indexing needs pack1 */
        if (q32.elempack != 1)
        {
            Mat q32_p1;
            convert_packing(q32, q32_p1, 1, opt);
            q32 = q32_p1;
        }

        std::vector<Mat> in32(3);
        in32[0] = q32;
        in32[1] = cos32;
        in32[2] = sin32;
        std::vector<Mat> out32(1);
        int ret = forward(in32, out32, opt); // fp32/pack1 now: normal path below
        if (ret != 0)
            return ret;

        Mat& top_blob = top_blobs[0];
        const Mat& t32 = out32[0];
        if (bottom_blob.elembits() == 16)
        {
            if (opt.use_bf16_storage)
                cast_float32_to_bfloat16(t32, top_blob, opt);
            else
                cast_float32_to_float16(t32, top_blob, opt);
        }
        else
            top_blob = t32;
        if (top_blob.elempack != bottom_blob.elempack)
        {
            Mat packed;
            convert_packing(top_blob, packed, bottom_blob.elempack, opt);
            top_blob = packed;
        }
        return 0;
    }

    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    /* input_hc_swapped=1：输入是 [w, c, h]（吸收掉前面的 Permute(order_type=2)），
     * 序列在 c 轴、头在 h 轴；输出仍写标准 [w, h, c]。 */
    const int seqlen = input_hc_swapped ? bottom_blob.c : bottom_blob.h;
    const int num_heads = input_hc_swapped ? bottom_blob.h : bottom_blob.c;

    Mat& top_blob = top_blobs[0];
    if (input_hc_swapped)
    {
        /* 融合模式下输入是 [w, c, h]，输出必须回到标准布局 [w, h, c]
         * （原来的 Permute(order_type=2) 干的活由本层一并完成）。 */
        top_blob.create(embed_dim, seqlen, num_heads, bottom_blob.elemsize, opt.blob_allocator);
    }
    else
    {
        top_blob.create_like(bottom_blob, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob.channel(q);
        Mat out_head = top_blob.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            if (interleaved)
            {
                /* 交换布局下 (头 q, 序列 i) 的元素在 channel(i).row(q) */
                const float* ptr = input_hc_swapped ? bottom_blob.channel(i).row(q) : head.row(i);
                const float* cos_ptr = cos_cache.row(i);
                const float* sin_ptr = sin_cache.row(i);
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
                const int half = embed_dim / 2;
                // A full-width (embed_dim) cache carries independent cos/sin for the two halves
                // (2D / vision rope where the halves differ); a half-width (embed_dim/2) cache is
                // the standard-rope form whose halves are identical, so the second half reuses the
                // first. This keeps existing half-width callers bit-identical.
                const int cw = cos_cache.w == embed_dim ? half : 0;
                // 交换布局下 (头 q, 序列 i) 的元素在 channel(i).row(q)
                const float* ptr0 = input_hc_swapped ? bottom_blob.channel(i).row(q) : head.row(i);
                const float* ptr1 = ptr0 + half;
                const float* cos_ptr0 = cos_cache.row(i);
                const float* sin_ptr0 = sin_cache.row(i);
                const float* cos_ptr1 = cos_ptr0 + cw;
                const float* sin_ptr1 = sin_ptr0 + cw;
                float* outptr0 = out_head.row(i);
                float* outptr1 = outptr0 + half;

                for (int j = 0; j < half; j++)
                {
                    const float x0 = ptr0[j];
                    const float x1 = ptr1[j];
                    outptr0[j] = x0 * cos_ptr0[j] - x1 * sin_ptr0[j];
                    outptr1[j] = x0 * sin_ptr1[j] + x1 * cos_ptr1[j];
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
