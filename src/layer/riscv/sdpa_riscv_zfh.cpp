// SpacemiT K3 IME2 SDPA acceleration
// SPDX-License-Identifier: BSD-3-Clause
//
// This TU is compiled with the zfh(+xsmtvdotii) march (see cmake/ncnn_add_layer.cmake)
// and contains the smt.vfwmadot assembly kernels; it is never executed when the
// runtime probe fails (non-A100).

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

    /* The pipeline-time probe only covers the context of that moment (binding can
     * change, the team can grow). Re-check before every forward that the current
     * team is on A100 right now; if not, return -99 so the dispatching side
     * falls back to the upstream GEMM path. */
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

    /* Back off on pathologically large scale: the ordinary fp16-packing rounding
     * error of Q/K (relative 2^-11) is amplified by scale until it is
     * significant in logit space (a difference in the 4th decimal of the keys
     * is indistinguishable in fp16 anyway — same as upstream fp16s storage
     * semantics; but only an extreme scale lets it affect the softmax
     * selection). Real attention scale ≈ 1/sqrt(E) ≤ 1, pathological cases
     * ≥ 1e4: threshold 1024 leaves over two orders of magnitude on both sides.
     * A finer value-based bound was tested: the worst-case bound for the real
     * model qwen3-0.6b on long prefill already reaches ~150 (mxq max 417, mxk
     * max 199), while its output is byte-identical to upstream — a value-based
     * bound cannot tell real inputs from adversarial ones; only scale can.
     * Hard out-of-range/underflow cases are handled by the range pre-check
     * below. */
    if (fabsf(_scale) > 1024.f)
        return -99;

    /* fp16 packing range pre-check: inputs with |v| > 65504 pack to ±inf while
     * the upstream fp32 path stays finite. This must be caught **before any
     * state change** (the kv cache append cannot be undone) so the dispatching
     * side can drop to upstream — when -99 is returned, top_blobs[1]/[2] have
     * not been written yet.
     * The magnitude of V is also a gate: the weight distortion caused by the
     * fp16 rounding collapse of Q/K (fp16 storage semantics, same as upstream
     * fp16s) is multiplied by V into the output — with pathologically large V
     * (e.g. a ±65504 cancellation construction), "averaging" and "selecting"
     * diverge hugely. Measured on the real model: mxv ≤ 63 (qwen3-0.6b, long
     * prefill, 28 calls); threshold 8192 leaves 130x headroom. */
    float mxv = 0.f;
    if (!ncnn_ime2_sdpa::ime2_fits_fp16(query) || !ncnn_ime2_sdpa::ime2_fits_fp16(cur_key)
            || !ncnn_ime2_sdpa::ime2_fits_fp16(cur_value, &mxv) || !ncnn_ime2_sdpa::ime2_fits_fp16(past_key)
            || !ncnn_ime2_sdpa::ime2_fits_fp16(past_value, &mxv))
        return -99;
    if (mxv > 8192.f)
        return -99;

    // ---- kv cache append (exactly the same semantics as the naive path) ----
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

    // ---- flash switch (on by default; NCNN_SDPA_FLASH=0 returns to the full-matrix old path, for A/B testing and as a fallback) ----
    int use_flash = 1;
    {
        const char* env = getenv("NCNN_SDPA_FLASH");
        if (env && env[0] == '0' && env[1] == '\0')
            use_flash = 0;
    }

    /* mask shape contract: consistent with the upstream generic path, attn_mask
     * must be a full [dst_seqlen, src_seqlen] (c=1 shared or per head);
     * broadcast shapes (1xN, Mx1, 1-D) are also indexed in full by row(i)
     * upstream, with no broadcast semantics. The flash path advances the mask
     * pointer row by row with stride ldm=dst_seqlen, and the causal scan also
     * takes rows by row(i) — both go out of bounds on non-full shapes. On
     * incomplete shapes, turn flash off and take the old path (whose row-wise
     * semantics are byte-identical to upstream); the causal scan is skipped
     * along with it. */
    if (use_flash
            && (attn_mask_blob.dims < 2 || attn_mask_blob.w < dst_seqlen || attn_mask_blob.h < src_seqlen))
        use_flash = 0;

    /* Reject oversized value heads BEFORE any flash packing: with
     * out_embed_dim > IME2_FLASH_MAXEV, ime2_flash_head_tiles returns -1
     * deterministically — but only after this branch has packed all shared
     * K/V and Q for every head, and the old path then packs them again.
     * Large-value-head models would pay both packing passes on every call. */
    if (use_flash && out_embed_dim > IME2_FLASH_MAXEV)
        use_flash = 0;

    // ---- causal mask detection (once per layer; if not causal, flash skips no blocks and adds the full mask, semantics unchanged) ----
    // 3-D masks are per head while this flag is applied to every head. Inspecting
    // only channel 0 means that if head 0 is triangular but some other head may
    // attend to a future position, the flash path truncates that head's KV scan
    // and the permitted values are never read. Every channel has to qualify.
    int causal = 1;
    if (use_flash)
    {
        /* All quantities of the truncation test are loop invariants:
         * margin/cutoff are computed once before the scan. Originally logf was
         * called per element in the inner j loop — the triangular region at
         * L=4096 means millions of libm calls, eating the long-prefill gain. */
        const float max_dot = (float)embed_dim * 65504.f * 65504.f;
        const int omitted = cur_seqlen - IME2_FLASH_BM;
        const float margin = 20.f + logf((float)(omitted > 1 ? omitted : 1));
        const float cutoff = -2.f * fabsf(_scale) * max_dot - margin;
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
                    /* Truncation requires every future logit to be below any
                     * visible logit; the one-sided "future is not positive"
                     * test is not enough (a visible logit can be more negative:
                     * when a visible K has the opposite sign of Q, the dot is
                     * negative, and scale·(-max_dot) + 0 can be even smaller
                     * than scale·(+max_dot) + v). Two-sided proof: visible
                     * logit ≥ -|scale|·max_dot (the visible region mask being
                     * all 0 was already guaranteed by the check above), future
                     * logit ≤ |scale|·max_dot + v; hence with
                     * v < -2·|scale|·max_dot the future is necessarily
                     * invisible. max_dot is the upper bound of the fp16 packing
                     * range, E·65504²; if the product overflows fp32 it becomes
                     * inf and the condition naturally fails (conservative
                     * direction). -inf is always hard. */
                    const float v = r[j];
                    if (v == -__builtin_inff())
                        continue;
                    /* One must also prove the softmax **mass** of future logits is
                     * negligible, not merely ordered lower:
                     * future ≤ |scale|·max_dot + v, visible row max ≥ -|scale|·max_dot,
                     * so the gap ≤ 2·|scale|·max_dot + v; then leave a margin of 20
                     * (exp(-20)≈2e-9, harmless mass). */
                    if (v < cutoff)
                        continue;
                    causal = 0;
                    break;
                }
            }
        }
    }

    if (use_flash)
    {
        // ---- K/V packing shared per GQA group (2:1 saves half the packing); Q packed per head, pre-scaled ----
        std::vector<Mat> KTg(num_group), VTg(num_group);
        int pack_fail = 0;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < num_group; g++)
        {
            if (ime2_pack_fp32(key.channel(g), embed_dim, KTg[g], dst_seqlen, embed_dim, opt.workspace_allocator) != 0)
                /* Workers write the same flag concurrently: an omp barrier cannot make
                                                                                                                                                                                                                                                                                                                                                                                 * the concurrent writes race-free, so use an atomic write. */
                #pragma omp atomic write
                pack_fail = 1;
            if (ime2_pack_fp32_T(value.channel(g), out_embed_dim, VTg[g], out_embed_dim, dst_seqlen, opt.workspace_allocator) != 0)
                /* Workers write the same flag concurrently: an omp barrier cannot make
                                                                                                                                                                                                                                                                                                                                                                                 * the concurrent writes race-free, so use an atomic write. */
                #pragma omp atomic write
                pack_fail = 1;
        }

        if (pack_fail)
            return -100;

        int flash_fail = 0;
        /* Validate INSIDE the execution region: OpenMP may select a different team
                         * for successive regions on a heterogeneous chip (e.g. per-worker affinity
                         * changed between the preflight and this region) — a worker validated as
                         * A100-bound earlier could now be X100-bound and would SIGILL on
                         * smt.vfwmadot. Same parallel region + barrier before any head runs. */
        #pragma omp parallel num_threads(opt.num_threads)
        {
            if (!ncnn_ime2_sdpa::sdpa_ime2_worker_in_set_now())
            {
                #pragma omp atomic write
                flash_fail = 1;
            }
            #pragma omp barrier
            if (!flash_fail)
            {
                #pragma omp for
                for (int q = 0; q < num_heads; q++)
                {
                    const Mat query_head = query.channel(q); // [L, E] fp32
                    Mat top_blob_head = top_blob.channel(q); // [L, Ev]
                    const int g = q / num_heads_per_group;
                    const Mat& maskm = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob;

                    Mat QT, PT;
                    /* Allocate once at the maximum capacity (incl. the tile slack): the per-block
                    * pack need never exceeds BM/8 x BN/8 x 64 + slack, so ime2_pack_fp32
                    * reuses this buffer instead of reallocating on every block. */
                    PT.create((size_t)(IME2_FLASH_BM / 8) * (IME2_FLASH_BN / 8) * 64 + IME2_TILE_SLACK, (size_t)2u, 1, opt.workspace_allocator);
                    if (PT.empty()
                            /* Q is not pre-multiplied by scale: fp16 packing would overflow Q·scale to inf; scale is applied
                                                                                                                                                                                                                                                                     * to the fp32 accumulation result inside the flash kernel (see ime2_flash_head_tiles). */
                            || ime2_pack_fp32(query_head, embed_dim, QT, src_seqlen, embed_dim, opt.workspace_allocator) != 0
                            /* ldm must be the mask's true row width maskm.w, not dst_seqlen:
                                                                                                                                                                                                                                                             * the guard allows wide masks with w >= dst (padded columns at row end); advancing
                                                                                                                                                                                                                                                             * by dst_seqlen would make row r read the previous row's values. */
                            || ime2_flash_head_tiles((const ime2_fp16*)QT, (const ime2_fp16*)KTg[g], (const ime2_fp16*)VTg[g],
                                                     maskm, maskm.w, (float*)top_blob_head, out_embed_dim,
                                                     src_seqlen, dst_seqlen, embed_dim, out_embed_dim, past_seqlen, causal, _scale,
                                                     PT, opt.workspace_allocator)
                            != 0)
                    {
                        /* Workers write the same flag concurrently: an omp barrier cannot make the concurrent writes race-free, so use an atomic write. */
                        #pragma omp atomic write
                        flash_fail = 1; // caller falls back to the old path as a whole (mixing the workspace allocations of both paths inside omp is unwise)
                    }
                } // omp for
            }     // if (!flash_fail) wrapper
        }         // omp parallel (validation + work in the same region)

        if (!flash_fail)
        {
            return 0;
        }
        // Some head failed: recompute from scratch with the old path as a whole (top_blob is overwritten, semantically safe)
    }

    // ---- Old path: per-head full QK^T -> mask -> softmax -> P·V ----
    Mat qk_cross(dst_seqlen, src_seqlen, opt.num_threads, 4u, opt.workspace_allocator);
    if (qk_cross.empty())
        return -100;

    int head_fail = 0;
    /* Validate INSIDE this region too: the preflight validated its own region's
     * workers, but this region may select a worker re-bound to X100 in between,
     * which would SIGILL on the unguarded smt.vfwmadot in ime2_gemm_fp32out.
     * team_bad -> return -99: the dispatching side recomputes the layer with the
     * upstream composed path (no IME2 at all). */
    int team_bad = 0;
    #pragma omp parallel num_threads(opt.num_threads)
    {
        if (!ncnn_ime2_sdpa::sdpa_ime2_worker_in_set_now())
        {
            #pragma omp atomic write
            team_bad = 1;
        }
        #pragma omp barrier
        if (!team_bad)
        {
            #pragma omp for
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
            /* Workers write the same flag concurrently: an omp barrier cannot make
                                                                                                                                                                                                                                                                                     * the concurrent writes race-free, so use an atomic write. */
            #pragma omp atomic write
            head_fail = 1;
            continue;
        }
        if (ime2_pack_fp32(key_head, embed_dim, BT, dst_seqlen, embed_dim, opt.workspace_allocator) != 0)
        {
            /* Workers write the same flag concurrently: an omp barrier cannot make
                                                                                                                                                                                                                                                                                     * the concurrent writes race-free, so use an atomic write. */
            #pragma omp atomic write
            head_fail = 1;
            continue;
        }

        // S = scale * Q·K^T
        ime2_gemm_fp32out((const ime2_fp16*)AT, (const ime2_fp16*)BT,
                          (float*)qk_cross_head, dst_seqlen,
                          src_seqlen, dst_seqlen, embed_dim, _scale);

        // mask (additive, -1e38 causal mask)
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

        // softmax (numerically safe: row-max subtraction)
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

        // O = P·V (V packed transposed). P is scaled up by 2^15 before packing: packing
        // small probabilities directly would underflow them to 0 and the aggregation
        // quality would be lost; gemm's alpha=1/32768 compensates on the fp32 result.
        {
            float* pp = (float*)qk_cross_head;
            const int nn = src_seqlen * dst_seqlen;
            int j = 0;
            while (j < nn)
            {
                const size_t vl = __riscv_vsetvl_e32m8(nn - j);
                __riscv_vse32_v_f32m8(pp + j, __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(pp + j, vl), 32768.f, vl), vl);
                j += (int)vl;
            }
        }
        Mat A2T, B2T;
        if (ime2_pack_fp32((const float*)qk_cross_head, dst_seqlen, A2T, src_seqlen, dst_seqlen, opt.workspace_allocator) != 0)
        {
            /* Workers write the same flag concurrently: an omp barrier cannot make
                                                                                                                                                                                                                                                                                     * the concurrent writes race-free, so use an atomic write. */
            #pragma omp atomic write
            head_fail = 1;
            continue;
        }
        if (ime2_pack_fp32_T((const float*)value_head, out_embed_dim, B2T, out_embed_dim, dst_seqlen, opt.workspace_allocator) != 0)
        {
            /* Workers write the same flag concurrently: an omp barrier cannot make
                                                                                                                                                                                                                                                                                     * the concurrent writes race-free, so use an atomic write. */
            #pragma omp atomic write
            head_fail = 1;
            continue;
        }

        ime2_gemm_fp32out((const ime2_fp16*)A2T, (const ime2_fp16*)B2T,
                          (float*)top_blob_head, out_embed_dim,
                          src_seqlen, out_embed_dim, dst_seqlen, 1.f / 32768.f);
    }
        }   // omp for
        }   // omp parallel (validation + work in the same region)

    if (team_bad)
        return -99;
    if (head_fail)
        return -100;

    return 0;
#endif // __riscv_v
}

} // namespace ncnn

#endif // NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
