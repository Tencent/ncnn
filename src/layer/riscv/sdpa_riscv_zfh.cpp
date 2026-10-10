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
#include <stdlib.h>
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

#if __riscv_v
static void ime2_sdpa_prof_dump(int L, int dst, int past)
{
    if (!getenv("NCNN_SDPA_PROF"))
        return;
    fprintf(stderr, "[sdpa-prof] L=%d dst=%d past=%d | QK=%.2fms softmax=%.2fms Ppack=%.2fms PV=%.2fms store=%.2fms\n",
            L, dst, past, ncnn_ime2_sdpa::g_prof_qk, ncnn_ime2_sdpa::g_prof_softmax,
            ncnn_ime2_sdpa::g_prof_ppack, ncnn_ime2_sdpa::g_prof_pv, ncnn_ime2_sdpa::g_prof_store);
    fprintf(stderr, "[sdpa-prof] QKVpack=%.2fms\n", ncnn_ime2_sdpa::g_prof_pack);
    ncnn_ime2_sdpa::g_prof_pack = 0;
    ncnn_ime2_sdpa::g_prof_qk = ncnn_ime2_sdpa::g_prof_softmax = 0;
    ncnn_ime2_sdpa::g_prof_ppack = ncnn_ime2_sdpa::g_prof_pv = ncnn_ime2_sdpa::g_prof_store = 0;
}
#endif

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

    // ---- flash 开关（默认开；NCNN_SDPA_FLASH=0 回到全量矩阵旧路径，供 A/B 与兜底） ----
    int use_flash = 1;
    {
        const char* env = getenv("NCNN_SDPA_FLASH");
        if (env && env[0] == '0' && env[1] == '\0')
            use_flash = 0;
    }

    // ---- causal 掩码检测（每层一次；非因果则 flash 不跳块、全量加 mask，语义不变） ----
    int causal = 1;
    {
        const Mat& mchk = attn_mask_blob.c > 1 ? attn_mask_blob.channel(0) : attn_mask_blob;
        for (int i = 0; i < src_seqlen && causal; i++)
        {
            const float* r = mchk.row(i);
            const int vis = past_seqlen + i + 1;
            for (int j = 0; j < vis && j < dst_seqlen; j++)
                if (r[j] != 0.f)
                {
                    causal = 0;
                    break;
                }
            for (int j = vis; j < dst_seqlen && causal; j++)
                if (r[j] > -1e30f)
                {
                    causal = 0;
                    break;
                }
        }
    }

    if (use_flash)
    {
        // ---- K/V 打包按 GQA 组共享（2:1 时省一半打包），Q 逐头预缩放打包 ----
        std::vector<Mat> KTg(num_group), VTg(num_group);
        const bool sdpa_prof = (getenv("NCNN_SDPA_PROF") != nullptr);
        const double t_pack0 = sdpa_prof ? ncnn_ime2_sdpa::ime2_prof_now() : 0.0;
        int pack_fail = 0;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < num_group; g++)
        {
            if (ime2_pack_fp32(key.channel(g), embed_dim, KTg[g], dst_seqlen, embed_dim, opt.workspace_allocator) != 0)
                pack_fail = 1;
            if (ime2_pack_fp32_T(value.channel(g), out_embed_dim, VTg[g], out_embed_dim, dst_seqlen, opt.workspace_allocator) != 0)
                pack_fail = 1;
        }
        const double t_pack1 = sdpa_prof ? ncnn_ime2_sdpa::ime2_prof_now() : 0.0;
        if (sdpa_prof)
        {
            #pragma omp atomic
            ncnn_ime2_sdpa::g_prof_pack += t_pack1 - t_pack0;
        }

        if (pack_fail)
            return -100;

        int flash_fail = 0;
        const double t_wall0 = sdpa_prof ? ncnn_ime2_sdpa::ime2_prof_now() : 0.0;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const Mat query_head = query.channel(q); // [L, E] fp32
            Mat top_blob_head = top_blob.channel(q); // [L, Ev]
            const int g = q / num_heads_per_group;
            const Mat& maskm = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob;

            Mat QT, PT;
            PT.create((size_t)(IME2_FLASH_BM / 8) * (IME2_FLASH_BN / 8) * 64, (size_t)2u, 1, opt.workspace_allocator);
            if (PT.empty()
                    || ime2_pack_fp32_scale(query_head, embed_dim, QT, src_seqlen, embed_dim, _scale, opt.workspace_allocator) != 0
                    || ime2_flash_head_tiles((const ime2_fp16*)QT, (const ime2_fp16*)KTg[g], (const ime2_fp16*)VTg[g],
                                             maskm, dst_seqlen, (float*)top_blob_head, out_embed_dim,
                                             src_seqlen, dst_seqlen, embed_dim, out_embed_dim, past_seqlen, causal,
                                             PT, opt.workspace_allocator)
                    != 0)
            {
                flash_fail = 1; // 调用方整体回退旧路径（omp 内不宜部分混跑两条路径的 workspace 分配）
            }
        }

        if (!flash_fail)
        {
            if (sdpa_prof)
                fprintf(stderr, "[sdpa-wall] QKVpack区=%.2fms heads区=%.2fms\n",
                        (t_pack1 - t_pack0), (ncnn_ime2_sdpa::ime2_prof_now() - t_wall0));
            ime2_sdpa_prof_dump(src_seqlen, dst_seqlen, past_seqlen);
            return 0;
        }
        // 有头失败：从头用旧路径整体重算（top_blob 会被覆盖写，语义安全）
    }

    // ---- 旧路径：per-head 全量 QK^T -> mask -> softmax -> P·V ----
    Mat qk_cross(dst_seqlen, src_seqlen, opt.num_threads, 4u, opt.workspace_allocator);
    if (qk_cross.empty())
        return -100;

    int head_fail = 0;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat query_head = query.channel(q);                       // [L, E] fp32
        const Mat key_head = key.channel(q / num_heads_per_group);     // [dst, E]
        const Mat value_head = value.channel(q / num_heads_per_group); // [dst, Ev]
        Mat qk_cross_head = qk_cross.channel(get_omp_thread_num());    // [L, dst]
        Mat top_blob_head = top_blob.channel(q);                       // [L, Ev]

        Mat AT, BT;
        if (ime2_pack_fp32(query_head, embed_dim, AT, src_seqlen, embed_dim, opt.workspace_allocator) != 0)
        {
            head_fail = 1;
            continue;
        }
        if (ime2_pack_fp32(key_head, embed_dim, BT, dst_seqlen, embed_dim, opt.workspace_allocator) != 0)
        {
            head_fail = 1;
            continue;
        }

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
        {
            head_fail = 1;
            continue;
        }
        if (ime2_pack_fp32_T((const float*)value_head, out_embed_dim, B2T, out_embed_dim, dst_seqlen, opt.workspace_allocator) != 0)
        {
            head_fail = 1;
            continue;
        }

        ime2_gemm_fp32out((const ime2_fp16*)A2T, (const ime2_fp16*)B2T,
                          (float*)top_blob_head, out_embed_dim,
                          src_seqlen, out_embed_dim, dst_seqlen, 1.f);
    }

    if (head_fail)
        return -100;

    ime2_sdpa_prof_dump(src_seqlen, dst_seqlen, past_seqlen);

    return 0;
#endif // __riscv_v
}

} // namespace ncnn

#endif // NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
