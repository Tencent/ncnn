// SpacemiT K3 IME2 SDPA acceleration
// SPDX-License-Identifier: BSD-3-Clause
//
// 本 TU 以 zfh(+xsmtvdotii) march 编译（见 cmake/ncnn_add_layer.cmake），
// 包含 smt.vfwmadot 汇编内核；运行时探测失败（非 A100）时绝不会执行到这里。

#include "sdpa_riscv.h"

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2

#include "platform.h"

#if __riscv_v
#include "sdpa_riscv_ime2.h"
#endif

#include <algorithm>
#include <float.h>
#include <math.h>
#include <string.h>

namespace ncnn {

int SDPA_riscv::ime2_available() const
{
#if !__riscv_v
    return 0;
#else
    return ncnn_ime2_sdpa::ime2_probe();
#endif
}

int SDPA_riscv::forward_ime2_prefill(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if !__riscv_v
    (void)bottom_blobs;
    (void)top_blobs;
    (void)opt;
    return -1;
#else
    using namespace ncnn_ime2_sdpa;

    const Mat& query = bottom_blobs[0];
    const Mat& cur_key = bottom_blobs[1];
    const Mat& cur_value = bottom_blobs[2];
    const Mat& attn_mask_blob = bottom_blobs[3];
    const Mat& past_key = bottom_blobs[4];
    const Mat& past_value = bottom_blobs[5];

    const int embed_dim = query.w;
    const int src_seqlen = query.h;
    const int num_heads = query.c;
    const int cur_seqlen = cur_key.h;
    const int num_group = cur_key.c;
    const int out_embed_dim = cur_value.w;
    const int past_seqlen = past_key.h;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    const float _scale = scale == 0.f ? 1.f / sqrt(embed_dim) : scale;
    const int num_heads_per_group = num_heads / num_group;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    Mat qk_cross(dst_seqlen, src_seqlen, opt.num_threads, 4u, opt.workspace_allocator);
    if (qk_cross.empty())
        return -100;

    // ---- kv cache 追加（与朴素路径完全相同的语义） ----
    Mat key = cur_key;
    Mat value = cur_value;
    {
        Mat& cached_key = top_blobs[1];
        Mat& cached_value = top_blobs[2];

        int retk = create_or_grow_kvcache(past_key, cached_key, dst_seqlen, num_group, embed_dim, cur_key.elemsize, cur_key.elempack, opt);
        if (retk != 0)
            return retk;
        int retv = create_or_grow_kvcache(past_value, cached_value, dst_seqlen, num_group, out_embed_dim, cur_value.elemsize, cur_value.elempack, opt);
        if (retv != 0)
            return retv;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_group; q++)
        {
            Mat key_head = cached_key.channel(q);
            Mat value_head = cached_value.channel(q);
            memcpy(key_head.row(past_seqlen), cur_key.channel(q), (size_t)embed_dim * cur_seqlen * cur_key.elemsize);
            memcpy(value_head.row(past_seqlen), cur_value.channel(q), (size_t)out_embed_dim * cur_seqlen * cur_value.elemsize);
        }

        key = cached_key;
        value = cached_value;
    }

    // ---- per-head: IME2 GEMM(QK^T) -> mask -> softmax -> IME2 GEMM(PV) ----
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat query_head = query.channel(q);                          // [L, E] fp32
        const Mat key_head = key.channel(q / num_heads_per_group);        // [dst, E]
        const Mat value_head = value.channel(q / num_heads_per_group);    // [dst, Ev]
        Mat qk_cross_head = qk_cross.channel(get_omp_thread_num());       // [L, dst]
        Mat top_blob_head = top_blob.channel(q);                          // [L, Ev]

        Mat AT, BT;
        if (ime2_pack_fp32(query_head, embed_dim, AT, src_seqlen, embed_dim, opt.workspace_allocator) != 0)
            continue; // 内存不足时静默跳过会破坏结果 —— 直接失败更好，但 omp 里只能记录；分配失败极罕见
        if (ime2_pack_fp32(key_head, embed_dim, BT, dst_seqlen, embed_dim, opt.workspace_allocator) != 0)
            continue;

        // S = scale * Q·K^T
        ime2_gemm_fp32out((const ime2_fp16*)AT, (const ime2_fp16*)BT,
                          (float*)qk_cross_head, dst_seqlen,
                          src_seqlen, dst_seqlen, embed_dim, _scale);

        // mask（加性，-1e38 因果掩码）
        {
            const Mat& maskm = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob;
            for (int i = 0; i < src_seqlen; i++)
            {
                const float* mptr = maskm.row(i);
                float* outptr = qk_cross_head.row(i);
                for (int j = 0; j < dst_seqlen; j++)
                    outptr[j] += mptr[j];
            }
        }

        // softmax（数值安全：行最大值减法）
        for (int i = 0; i < src_seqlen; i++)
        {
            float* ptr = qk_cross_head.row(i);
            float max = -FLT_MAX;
            for (int j = 0; j < dst_seqlen; j++)
                max = std::max(max, ptr[j]);
            float sum = 0.f;
            for (int j = 0; j < dst_seqlen; j++)
            {
                ptr[j] = (float)expf(ptr[j] - max);
                sum += ptr[j];
            }
            for (int j = 0; j < dst_seqlen; j++)
                ptr[j] /= sum;
        }

        // O = P·V（V 转置打包）
        Mat A2T, B2T;
        if (ime2_pack_fp32((const float*)qk_cross_head, dst_seqlen, A2T, src_seqlen, dst_seqlen, opt.workspace_allocator) != 0)
            continue;
        if (ime2_pack_fp32_T((const float*)value_head, out_embed_dim, B2T, out_embed_dim, dst_seqlen, opt.workspace_allocator) != 0)
            continue;

        ime2_gemm_fp32out((const ime2_fp16*)A2T, (const ime2_fp16*)B2T,
                          (float*)top_blob_head, out_embed_dim,
                          src_seqlen, out_embed_dim, dst_seqlen, 1.f);
    }

    return 0;
#endif // __riscv_v
}

} // namespace ncnn

#endif // NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
