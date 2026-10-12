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

int SDPA_riscv::ime2_available(int num_threads) const
{
#if !__riscv_v
    (void)num_threads;
    return 0;
#else
    return ncnn_ime2_sdpa::ime2_probe(num_threads);
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

    /* 建管线探测只覆盖当时上下文（绑定可改、团队可变大）。每次 forward 前校验
     * 当前团队此刻都在 A100 上；不满足返回 -99，调度方回落上游 GEMM 路径。 */
    if (!ncnn_ime2_sdpa::ime2_team_ok(opt.num_threads))
        return -99;

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

    /* fp16 打包值域预检：|v| > 65504 的输入打包成 ±inf，而上游 fp32 路径保持
     * 有限。必须在**任何状态改动之前**（kv cache 追加不可回退）发现并让
     * 调度方落上游——返回 -99 时 top_blobs[1]/[2] 尚未写入。 */
    if (!ncnn_ime2_sdpa::ime2_fits_fp16(query) || !ncnn_ime2_sdpa::ime2_fits_fp16(cur_key)
            || !ncnn_ime2_sdpa::ime2_fits_fp16(cur_value) || !ncnn_ime2_sdpa::ime2_fits_fp16(past_key)
            || !ncnn_ime2_sdpa::ime2_fits_fp16(past_value))
        return -99;

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

    /* mask 形状契约：与上游通用路径一致，attn_mask 必须是完整的
     * [dst_seqlen, src_seqlen]（c=1 共享或逐头）；广播形状（1xN、Mx1、1-D）
     * 在上游同样按 row(i) 全量索引，没有广播语义。flash 路径按行距
     * ldm=dst_seqlen 逐行推进 mask 指针，causal 扫描也按 row(i) 取行——对
     * 非完整形状两者都会越界。形状不完整时关掉 flash 走旧路径（其逐行语义与
     * 上游逐字节一致），causal 扫描也随之跳过。 */
    if (use_flash
            && (attn_mask_blob.dims < 2 || attn_mask_blob.w < dst_seqlen || attn_mask_blob.h < src_seqlen))
        use_flash = 0;

    // ---- causal 掩码检测（每层一次；非因果则 flash 不跳块、全量加 mask，语义不变） ----
    // 3-D masks are per head while this flag is applied to every head. Inspecting
    // only channel 0 means that if head 0 is triangular but some other head may
    // attend to a future position, the flash path truncates that head's KV scan
    // and the permitted values are never read. Every channel has to qualify.
    int causal = 1;
    if (use_flash)
    {
        const int nmask = attn_mask_blob.c > 1 ? attn_mask_blob.c : 1;
        for (int g = 0; g < nmask && causal; g++)
        {
            const Mat& mchk = attn_mask_blob.c > 1 ? attn_mask_blob.channel(g) : attn_mask_blob;
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
                {
                    /* 只有**确实不可逾越**的掩码才算因果屏障：-inf 永远成立；
                     * 有限大负值（如 ncnn_llm 用的 -1e38）只在 scale 无法翻转
                     * 符号时成立——|q·k| ≤ E·65504²（fp16 打包值域），故
                     * |scale| < 1e38/(E·65504²) 时 -1e38 必为硬屏障；病态大
                     * scale 下有限负值是软掩码，放弃因果判定全量加 mask
                     * （保守但正确，仅损失跳块收益）。 */
                    const float v = r[j];
                    if (v == -__builtin_inff())
                        continue;
                    const float max_dot = (float)embed_dim * 65504.f * 65504.f;
                    if (fabsf(_scale) < 1e38f / max_dot && v <= -1e38f)
                        continue;
                    causal = 0;
                    break;
                }
            }
        }
    }

    if (use_flash)
    {
        // ---- K/V 打包按 GQA 组共享（2:1 时省一半打包），Q 逐头预缩放打包 ----
        std::vector<Mat> KTg(num_group), VTg(num_group);
        int pack_fail = 0;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < num_group; g++)
        {
            if (ime2_pack_fp32(key.channel(g), embed_dim, KTg[g], dst_seqlen, embed_dim, opt.workspace_allocator) != 0)
                /* 工人并发写同一个标志：omp 屏障不能让并发写变成无竞争的，用原子写。 */
                #pragma omp atomic write
                pack_fail = 1;
            if (ime2_pack_fp32_T(value.channel(g), out_embed_dim, VTg[g], out_embed_dim, dst_seqlen, opt.workspace_allocator) != 0)
                /* 工人并发写同一个标志：omp 屏障不能让并发写变成无竞争的，用原子写。 */
                #pragma omp atomic write
                pack_fail = 1;
        }

        if (pack_fail)
            return -100;

        int flash_fail = 0;
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
                    /* Q 不预乘 scale：fp16 打包会把 Q·scale 溢出成 inf；scale 在
                                                     * flash 内核里施加于 fp32 累加结果（见 ime2_flash_head_tiles）。 */
                    || ime2_pack_fp32(query_head, embed_dim, QT, src_seqlen, embed_dim, opt.workspace_allocator) != 0
                    /* ldm 必须是 mask 的真实行宽 maskm.w，而不是 dst_seqlen：
                                             * 守卫允许 w >= dst 的宽 mask（行尾有填充列），按 dst_seqlen
                                             * 推进会让第 r 行读到前一行的值。 */
                    || ime2_flash_head_tiles((const ime2_fp16*)QT, (const ime2_fp16*)KTg[g], (const ime2_fp16*)VTg[g],
                                             maskm, maskm.w, (float*)top_blob_head, out_embed_dim,
                                             src_seqlen, dst_seqlen, embed_dim, out_embed_dim, past_seqlen, causal, _scale,
                                             PT, opt.workspace_allocator)
                    != 0)
            {
                /* 工人并发写同一标志：omp 屏障不能让并发写变成无竞争，用原子写。 */
                #pragma omp atomic write
                flash_fail = 1; // 调用方整体回退旧路径（omp 内不宜部分混跑两条路径的 workspace 分配）
            }
        }

        if (!flash_fail)
        {
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
            /* 工人并发写同一个标志：omp 屏障不能让并发写变成无竞争的，用原子写。 */
            #pragma omp atomic write
            head_fail = 1;
            continue;
        }
        if (ime2_pack_fp32(key_head, embed_dim, BT, dst_seqlen, embed_dim, opt.workspace_allocator) != 0)
        {
            /* 工人并发写同一个标志：omp 屏障不能让并发写变成无竞争的，用原子写。 */
            #pragma omp atomic write
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
            /* 工人并发写同一个标志：omp 屏障不能让并发写变成无竞争的，用原子写。 */
            #pragma omp atomic write
            head_fail = 1;
            continue;
        }
        if (ime2_pack_fp32_T((const float*)value_head, out_embed_dim, B2T, out_embed_dim, dst_seqlen, opt.workspace_allocator) != 0)
        {
            /* 工人并发写同一个标志：omp 屏障不能让并发写变成无竞争的，用原子写。 */
            #pragma omp atomic write
            head_fail = 1;
            continue;
        }

        ime2_gemm_fp32out((const ime2_fp16*)A2T, (const ime2_fp16*)B2T,
                          (float*)top_blob_head, out_embed_dim,
                          src_seqlen, out_embed_dim, dst_seqlen, 1.f);
    }

    if (head_fail)
        return -100;

    return 0;
#endif // __riscv_v
}

} // namespace ncnn

#endif // NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
