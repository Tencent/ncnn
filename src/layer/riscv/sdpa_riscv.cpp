// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_riscv.h"

#include "layer_type.h"

#include <cstdlib>
#include <cstdio>
#include <float.h>
#include <math.h>
#include <string.h>
#include <time.h>

#if __riscv_v
#include <riscv_vector.h>
#endif
#if defined __linux__
#include <sched.h>
#endif

namespace ncnn {

SDPA_riscv::SDPA_riscv()
{
    qk_gemm = 0;
    qkv_gemm = 0;
    qk_softmax = 0;
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    use_ime2_sdpa = 0;
#endif
}

#if __riscv_v
// ---------------- RVV decode helpers (M=1 attention, f32) ----------------
// All vsetvl calls are explicit (at VLEN=1024 the VL cap of e32m8 is 256; hard-coding zero would go out of bounds).

#if __riscv_vector
// Vectorized exp (same implementation as swish_riscv.cpp): exp(x) = 2^(x*log2e) = 2^n * 2^f
// Element-wise arithmetic, result is VLEN-agnostic.
// Note the boundary of cross-cluster byte-exactness: exp and this layer's softmax
// summation (fixed 8 lanes + vfredosum) are VLEN-agnostic; but rvv_dot_f32 (QK)
// and the PV accumulate are still grouped by lane (the e32m8 width varies with
// VLEN), so the whole decode path is byte-reproducible only on the **same
// cluster** — on K3 the process is pinned inside the A100 cluster after
// registration, so this premise holds; consistent with ncnn globally (the
// summation order of packing tiles varies with VLEN anyway, and upstream never
// promises bit-identical results across VLEN).
// The decode softmax originally used scalar expf (measured 44.7 ns/element);
// at 507 tokens that is 8112 calls per layer ≈ 363 µs, the main cost of
// long-context decode (about 10 ms per token). Here it is replaced with the vector version.
static inline vfloat32m1_t sdpa_vexp_f32m1(vfloat32m1_t x, size_t vl)
{
    /* Bit-exact replica of the exp_ps (rvv_mathfun.h) construction used by
     * upstream Softmax_riscv: same cephes polynomial, same clamping constants,
     * same +0.5/floor rounding, same ldexp bit trick. The decode fast-path
     * output is bit-identical to the composed fallback path; values below
     * exp_lo are flushed to 0 as upstream does (NaN clamping semantics also
     * match upstream: vfmin/vfmax with a single NaN take the numeric operand). */
    /* Constant names get an sdpa_ prefix: these are macros in rvv_mathfun.h, and same-named local declarations would clash. */
    const float sdpa_exp_hi = 88.3762626647949f;
    const float sdpa_exp_lo = -88.3762626647949f; /* actual clamping lower bound of upstream exp_ps */
    const float sdpa_cephes_LOG2EF = 1.44269504088896341f;
    const float sdpa_cephes_exp_C1 = 0.693359375f;
    const float sdpa_cephes_exp_C2 = -2.12194440e-4f;
    const float sdpa_cephes_exp_p0 = 1.9875691500E-4f;
    const float sdpa_cephes_exp_p1 = 1.3981999507E-3f;
    const float sdpa_cephes_exp_p2 = 8.3334519073E-3f;
    const float sdpa_cephes_exp_p3 = 4.1665795894E-2f;
    const float sdpa_cephes_exp_p4 = 1.6666665459E-1f;
    const float sdpa_cephes_exp_p5 = 5.0000001201E-1f;

    vfloat32m1_t tmp, fx;

    x = __riscv_vfmin_vf_f32m1(x, sdpa_exp_hi, vl);
    x = __riscv_vfmax_vf_f32m1(x, sdpa_exp_lo, vl);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = __riscv_vfmacc_vf_f32m1(__riscv_vfmv_v_f_f32m1(0.5f, vl), sdpa_cephes_LOG2EF, x, vl);

    /* floorf: convert to int and back; subtract 1 if greater than fx (verbatim copy of the upstream floor emulation) */
    tmp = __riscv_vfcvt_f_x_v_f32m1(__riscv_vfcvt_x_f_v_i32m1(fx, vl), vl);
    vbool32_t mask = __riscv_vmfgt_vv_f32m1_b32(tmp, fx, vl);
    fx = __riscv_vfsub_vf_f32m1_mu(mask, tmp, tmp, 1.f, vl);

    tmp = __riscv_vfmul_vf_f32m1(fx, sdpa_cephes_exp_C1, vl);
    vfloat32m1_t z = __riscv_vfmul_vf_f32m1(fx, sdpa_cephes_exp_C2, vl);
    x = __riscv_vfsub_vv_f32m1(x, tmp, vl);
    x = __riscv_vfsub_vv_f32m1(x, z, vl);

    vfloat32m1_t y = __riscv_vfmul_vf_f32m1(x, sdpa_cephes_exp_p0, vl);
    z = __riscv_vfmul_vv_f32m1(x, x, vl);
    y = __riscv_vfadd_vf_f32m1(y, sdpa_cephes_exp_p1, vl);
    y = __riscv_vfmul_vv_f32m1(y, x, vl);
    y = __riscv_vfadd_vf_f32m1(y, sdpa_cephes_exp_p2, vl);
    y = __riscv_vfmul_vv_f32m1(y, x, vl);
    y = __riscv_vfadd_vf_f32m1(y, sdpa_cephes_exp_p3, vl);
    y = __riscv_vfmul_vv_f32m1(y, x, vl);
    y = __riscv_vfadd_vf_f32m1(y, sdpa_cephes_exp_p4, vl);
    y = __riscv_vfmul_vv_f32m1(y, x, vl);
    y = __riscv_vfadd_vf_f32m1(y, sdpa_cephes_exp_p5, vl);
    /* Match upstream: separate multiply-add (two roundings), not FMA — otherwise the last bit differs by 1 ulp. */
    y = __riscv_vfmul_vv_f32m1(y, z, vl);
    y = __riscv_vfadd_vv_f32m1(y, x, vl);
    y = __riscv_vfadd_vf_f32m1(y, 1.f, vl);

    /* build 2^n */
    vint32m1_t mm = __riscv_vfcvt_x_f_v_i32m1(fx, vl);
    mm = __riscv_vadd_vx_i32m1(mm, 0x7f, vl);
    mm = __riscv_vsll_vx_i32m1(mm, 23, vl);
    vfloat32m1_t pow2n = __riscv_vreinterpret_v_i32m1_f32m1(mm);
    y = __riscv_vfmul_vv_f32m1(y, pow2n, vl);
    return y;
}

static inline vfloat32m8_t sdpa_vexp_f32m8(vfloat32m8_t x, size_t vl)
{
    /* Bit-exact replica of the exp_ps (rvv_mathfun.h) construction used by
     * upstream Softmax_riscv: same cephes polynomial, same clamping constants,
     * same +0.5/floor rounding, same ldexp bit trick. The decode fast-path
     * output is bit-identical to the composed fallback path; values below
     * exp_lo are flushed to 0 as upstream does (NaN clamping semantics also
     * match upstream: vfmin/vfmax with a single NaN take the numeric operand). */
    /* Constant names get an sdpa_ prefix: these are macros in rvv_mathfun.h, and same-named local declarations would clash. */
    const float sdpa_exp_hi = 88.3762626647949f;
    const float sdpa_exp_lo = -88.3762626647949f; /* actual clamping lower bound of upstream exp_ps */
    const float sdpa_cephes_LOG2EF = 1.44269504088896341f;
    const float sdpa_cephes_exp_C1 = 0.693359375f;
    const float sdpa_cephes_exp_C2 = -2.12194440e-4f;
    const float sdpa_cephes_exp_p0 = 1.9875691500E-4f;
    const float sdpa_cephes_exp_p1 = 1.3981999507E-3f;
    const float sdpa_cephes_exp_p2 = 8.3334519073E-3f;
    const float sdpa_cephes_exp_p3 = 4.1665795894E-2f;
    const float sdpa_cephes_exp_p4 = 1.6666665459E-1f;
    const float sdpa_cephes_exp_p5 = 5.0000001201E-1f;

    vfloat32m8_t tmp, fx;

    x = __riscv_vfmin_vf_f32m8(x, sdpa_exp_hi, vl);
    x = __riscv_vfmax_vf_f32m8(x, sdpa_exp_lo, vl);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = __riscv_vfmacc_vf_f32m8(__riscv_vfmv_v_f_f32m8(0.5f, vl), sdpa_cephes_LOG2EF, x, vl);

    /* floorf: convert to int and back; subtract 1 if greater than fx (verbatim copy of the upstream floor emulation) */
    tmp = __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8(fx, vl), vl);
    vbool4_t mask = __riscv_vmfgt_vv_f32m8_b4(tmp, fx, vl);
    fx = __riscv_vfsub_vf_f32m8_mu(mask, tmp, tmp, 1.f, vl);

    tmp = __riscv_vfmul_vf_f32m8(fx, sdpa_cephes_exp_C1, vl);
    vfloat32m8_t z = __riscv_vfmul_vf_f32m8(fx, sdpa_cephes_exp_C2, vl);
    x = __riscv_vfsub_vv_f32m8(x, tmp, vl);
    x = __riscv_vfsub_vv_f32m8(x, z, vl);

    vfloat32m8_t y = __riscv_vfmul_vf_f32m8(x, sdpa_cephes_exp_p0, vl);
    z = __riscv_vfmul_vv_f32m8(x, x, vl);
    y = __riscv_vfadd_vf_f32m8(y, sdpa_cephes_exp_p1, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl);
    y = __riscv_vfadd_vf_f32m8(y, sdpa_cephes_exp_p2, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl);
    y = __riscv_vfadd_vf_f32m8(y, sdpa_cephes_exp_p3, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl);
    y = __riscv_vfadd_vf_f32m8(y, sdpa_cephes_exp_p4, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl);
    y = __riscv_vfadd_vf_f32m8(y, sdpa_cephes_exp_p5, vl);
    /* Match upstream: separate multiply-add (two roundings), not FMA — otherwise the last bit differs by 1 ulp. */
    y = __riscv_vfmul_vv_f32m8(y, z, vl);
    y = __riscv_vfadd_vv_f32m8(y, x, vl);
    y = __riscv_vfadd_vf_f32m8(y, 1.f, vl);

    /* build 2^n */
    vint32m8_t mm = __riscv_vfcvt_x_f_v_i32m8(fx, vl);
    mm = __riscv_vadd_vx_i32m8(mm, 0x7f, vl);
    mm = __riscv_vsll_vx_i32m8(mm, 23, vl);
    vfloat32m8_t pow2n = __riscv_vreinterpret_v_i32m8_f32m8(mm);
    y = __riscv_vfmul_vv_f32m8(y, pow2n, vl);
    return y;
}
#endif

#if __riscv_vector
// Local clock for decode-attention segment timing (does not depend on the IME2-side profiler, so all variants compile)
#include <time.h>
static inline double sdpa_dec_now()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
#endif

static inline float rvv_dot_f32(const float* a, const float* b, int n)
{
    /* n can be far larger than VLMAX: on a VLEN=256 part e32m8 has 64 lanes, so
     * head_dim=128 needs two chunks. Reducing only the final chunk's vl would
     * discard the lanes last_vl..VLMAX-1 that earlier chunks accumulated into,
     * so each chunk is reduced to a scalar and the scalars are added. */
    /* One accumulator vector across chunks, in k order: per-lane partial sums
     * cancel large opposing terms that collide in the same lane (e.g. 1e20 at k=0
     * and -1e20 at k=VLMAX both land in lane 0 and cancel exactly), matching the
     * fallback gemm_transB_packed_tile's in-order per-element accumulation. The
     * previous per-chunk scalar reduction rounded each chunk separately and lost
     * such cancellations. _tu keeps earlier chunks' lanes on the tail chunk. */
    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.f, vlmax);
    int k = 0;
    while (k < n)
    {
        const size_t vl = __riscv_vsetvl_e32m8(n - k);
        acc = __riscv_vfmacc_vv_f32m8_tu(acc, __riscv_vle32_v_f32m8(a + k, vl), __riscv_vle32_v_f32m8(b + k, vl), vl);
        k += (int)vl;
    }
    /* Unordered reduction (vfredusum), matching upstream byte-for-byte on the
     * real workloads: an ordered reduction (vfredosum) changed the output md5
     * vs upstream on the reference prompt (upstream's own M=1 dot is itself not
     * kk-sequential as the tile-kernel reading suggested). The single-accumulator
     * cross-chunk structure above preserves the cancellation Codex demonstrated;
     * keep the upstream-matching unordered finish. */
    vfloat32m1_t s = __riscv_vfredusum_vs_f32m8_f32m1(acc, __riscv_vfmv_v_f_f32m1(0.f, 1), vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(s);
}

static inline float rvv_max_f32(const float* a, int n)
{
    vfloat32m1_t vm = __riscv_vfmv_v_f_f32m1(-FLT_MAX, 1);
    int k = 0;
    while (k < n)
    {
        size_t vl = __riscv_vsetvl_e32m8(n - k);
        vm = __riscv_vfredmax_vs_f32m8_f32m1(__riscv_vle32_v_f32m8(a + k, vl), vm, vl);
        k += (int)vl;
    }
    return __riscv_vfmv_f_s_f32m1_f32(vm);
}
#endif // __riscv_v

int SDPA_riscv::forward_rvv_decode(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if !__riscv_v
    (void)bottom_blobs;
    (void)top_blobs;
    (void)opt;
    return -1;
#else
    const Mat& query = bottom_blobs[0];
    const Mat& cur_key = bottom_blobs[1];
    const Mat& cur_value = bottom_blobs[2];
    const Mat& attn_mask_blob = bottom_blobs[3];
    const Mat& past_key = bottom_blobs[4];
    const Mat& past_value = bottom_blobs[5];

    if (query.elembits() != 32 || query.elempack != 1 || cur_key.elempack != 1 || cur_value.elempack != 1)
        return -1;

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

    // One scores scratch per thread. Must be allocated **before touching the KV cache**:
    // once the cache has been appended, falling back to the naive path is no longer possible (it would append twice).
    const int nthreads = opt.num_threads > 0 ? opt.num_threads : 1;
    /* Mat with a null workspace_allocator falls back to the default allocator,
     * so a bare layer call (e.g. tests/test_sdpa_kvcache.cpp, which uses a
     * default ncnn::Option) works instead of dereferencing null. The Mat also
     * frees itself, which removes the four manual fastFree early-return paths. */
    Mat score_slab_mat(dst_seqlen, nthreads, 4u, opt.workspace_allocator);
    if (score_slab_mat.empty())
        return -100;
    float* score_slab = score_slab_mat;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty())
    {
        return -100;
    }

    // ---- kv cache append (byte-identical semantics to the naive path) ----
    Mat& cached_key = top_blobs[1];
    Mat& cached_value = top_blobs[2];

    int retk = create_or_grow_kvcache(past_key, cached_key, dst_seqlen, num_group, embed_dim, cur_key.elemsize, cur_key.elempack, opt);
    if (retk != 0)
    {
        return retk;
    }
    int retv = create_or_grow_kvcache(past_value, cached_value, dst_seqlen, num_group, out_embed_dim, cur_value.elemsize, cur_value.elempack, opt);
    if (retv != 0)
    {
        return retv;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < num_group; g++)
    {
        Mat key_head = cached_key.channel(g);
        Mat value_head = cached_value.channel(g);
        memcpy(key_head.row(past_seqlen), cur_key.channel(g), (size_t)embed_dim * cur_seqlen * cur_key.elemsize);
        memcpy(value_head.row(past_seqlen), cur_value.channel(g), (size_t)out_embed_dim * cur_seqlen * cur_value.elemsize);
    }

    // ---- per-head attention ----
    /* Per-worker VLEN validation INSIDE the execution region: with per-thread
     * bindings (OMP_PROC_BIND/GOMP_CPU_AFFINITY) a worker may sit on an X100
     * (e32m8 VLMAX 64) even when the dispatching thread's affinity mask says
     * A100 only -- rvv_dot_f32 would go multi-chunk there for E > 64 and
     * reorder the dot product. The static mask gate above cannot see disjoint
     * worker masks; this check reads each actual worker's own vlenb-derived
     * VLMAX (one instruction, no syscall). team_vl_bad is written ONLY before
     * the barrier so the guard is immutable there; a failure falls back to the
     * composed path via -99. */
    int team_vl_bad = 0;
    #pragma omp parallel num_threads(opt.num_threads)
    {
        if (embed_dim > (int)__riscv_vsetvlmax_e32m8())
        {
            /* Workers write the same flag concurrently: an omp barrier cannot make
             * the concurrent writes race-free, so use an atomic write. */
            #pragma omp atomic write
            team_vl_bad = 1;
        }
        #pragma omp barrier
        if (!team_vl_bad)
        {
            #pragma omp for
            for (int q = 0; q < num_heads; q++)
            {
        const Mat query_head = query.channel(q); // [E, 1]
        const Mat key_head = cached_key.channel(q / num_heads_per_group);
        const Mat value_head = cached_value.channel(q / num_heads_per_group);
        Mat top_blob_head = top_blob.channel(q);

        const float* qptr = query_head.row(0);
        float* outptr = top_blob_head.row(0);
        const Mat& maskm = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob;
        const float* mptr = maskm.row(0);

        float* scores = score_slab + (size_t)get_omp_thread_num() * dst_seqlen;

        const bool dprof = (getenv("NCNN_SDPA_DECODE_PROF") != nullptr);
        const double dt0 = dprof ? sdpa_dec_now() : 0.0;

        // QK^T: scores[j] = (q · k_j) * scale + mask[j]
        for (int j = 0; j < dst_seqlen; j++)
        {
            const float* kptr = key_head.row(j);
            scores[j] = rvv_dot_f32(qptr, kptr, embed_dim) * _scale + mptr[j];
        }

        const double dt1 = dprof ? sdpa_dec_now() : 0.0;

        // softmax (same order as the naive path: max → expf → sum → multiply by reciprocal)
        const float mx = rvv_max_f32(scores, dst_seqlen);
        float sum = 0.f;
#if __riscv_vector
        /* Vectorized exp (was per-element expf calls, 44.7 ns/element → ~363 µs per layer at 507 tokens) */
        {
            /* The summation grouping must be VLEN-agnostic: the lane count
             * follows VLEN (A100 e32m1 = 32 lanes, X100 = 16 lanes), so
             * lane-grouped accumulate + unordered reduction would make sum
             * differ across clusters. Fixed 8-lane grouping (every supported
             * VLEN has VLMAX≥8), reduction via vfredosum (ordered, fixed
             * sequential semantics) — byte-identical on both clusters. The
             * accumulate structure of exp itself matches upstream
             * Softmax_riscv verbatim (e32m8 bulk groups + _tu tail group +
             * vfredusum finish): if the small terms all landed on the lane
             * holding the maximum they would be dropped one by one
             * (1 + 2.5e-8 == 1); bulk-group striding spreads the small terms
             * across independent per-lane accumulators, which preserves the
             * aggregation quality — the premise of bit-parity with the
             * fallback path. */
            vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvlmax_e32m8());
            {
                const size_t vlfull = __riscv_vsetvlmax_e32m8();
                float* ptr = scores;
                int n = dst_seqlen / (int)vlfull * (int)vlfull;
                while (n > 0)
                {
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(ptr, vlfull);
                    v = __riscv_vfsub_vf_f32m8(v, mx, vlfull);
                    /* m8 decode exp: the wide version with the same construction as exp_ps (below) */
                    v = sdpa_vexp_f32m8(v, vlfull);
                    __riscv_vse32_v_f32m8(ptr, v, vlfull);
                    vsum = __riscv_vfadd_vv_f32m8(vsum, v, vlfull);
                    ptr += vlfull;
                    n -= (int)vlfull;
                }
                int remain = dst_seqlen % (int)vlfull;
                if (remain > 0)
                {
                    const size_t vlr = __riscv_vsetvl_e32m8(remain);
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(ptr, vlr);
                    v = __riscv_vfsub_vf_f32m8(v, mx, vlr);
                    v = sdpa_vexp_f32m8(v, vlr);
                    __riscv_vse32_v_f32m8(ptr, v, vlr);
                    vsum = __riscv_vfadd_vv_f32m8_tu(vsum, vsum, v, vlr);
                }
                /* Same as upstream: vfredusum (unordered but deterministic on the same implementation) */
                sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(vsum, __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1()), __riscv_vsetvlmax_e32m8()));
            }
        }
#else
        for (int j = 0; j < dst_seqlen; j++)
        {
            scores[j] = (float)expf(scores[j] - mx);
            sum += scores[j];
        }
#endif
        const float inv_sum = 1.f / sum;
        for (int j = 0; j < dst_seqlen; j++)
            scores[j] *= inv_sum;

        const double dt2 = dprof ? sdpa_dec_now() : 0.0;

        // P·V: out[e] += p_j * v_j[e]; the output row stays resident in registers.
        // Note: whether the whole row can be written in one pass must be judged by
        // the **VLMAX** of e32m8 (varies with VLEN: VLEN=1024 → 256, VLEN=256 → 64);
        // otherwise on small-VLEN clusters like X100 only the first vl elements get
        // written and the rest stay uninitialized → garbage output (this bug was
        // masked by the "the X100 leg never actually took effect" gate).
        if (out_embed_dim <= (int)__riscv_vsetvlmax_e32m8())
        {
            size_t vl = __riscv_vsetvl_e32m8(out_embed_dim);
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.f, vl);
            for (int j = 0; j < dst_seqlen; j++)
            {
                const float* vptr = value_head.row(j);
                acc = __riscv_vfmacc_vf_f32m8(acc, scores[j], __riscv_vle32_v_f32m8(vptr, vl), vl);
            }
            __riscv_vse32_v_f32m8(outptr, acc, vl);
        }
        else
        {
            for (int e = 0; e < out_embed_dim; e++)
                outptr[e] = 0.f;
            for (int j = 0; j < dst_seqlen; j++)
            {
                const float* vptr = value_head.row(j);
                const float p = scores[j];
                int e = 0;
                while (e < out_embed_dim)
                {
                    size_t vl = __riscv_vsetvl_e32m8(out_embed_dim - e);
                    vfloat32m8_t acc = __riscv_vle32_v_f32m8(outptr + e, vl);
                    acc = __riscv_vfmacc_vf_f32m8(acc, p, __riscv_vle32_v_f32m8(vptr + e, vl), vl);
                    __riscv_vse32_v_f32m8(outptr + e, acc, vl);
                    e += (int)vl;
                }
            }
        }

        if (dprof && q < 2)
        {
            const double dt3 = sdpa_dec_now();
            #pragma omp critical
            {
                static int pc = 0;
                if (pc++ < 6)
                    fprintf(stderr, "[dec-prof] head=%d dst=%d | QK=%.3fms softmax=%.3fms PV=%.3fms\n",
                            q, dst_seqlen, dt1 - dt0, dt2 - dt1, dt3 - dt2);
            }
        }
    }
        }   // omp for
    }     // omp parallel (per-worker VLEN validation + work in the same region)

    if (team_vl_bad)
        return -99;

    return 0;
#endif // __riscv_v
}

int SDPA_riscv::create_pipeline(const Option& _opt)
{
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    // Debug switch: NCNN_SDPA_IME2=0 forces it off (for problem triage). Must come
    // before the probe: otherwise even the forced-off case still pays for one
    // per-core affinity sweep plus the process-level SIGILL probe.
    const char* ime2env = getenv("NCNN_SDPA_IME2");
    if (ime2env && ime2env[0] == '0' && ime2env[1] == '\0')
    {
        use_ime2_sdpa = 0;
    }
    else
    {
        // The IME2 probe is done in the zfh variant TU (which has the smt.vfwmadot assembly)
        // Probe with the same team size as forward: forward's parallel regions all use num_threads(opt.num_threads)
        use_ime2_sdpa = ime2_available(_opt.num_threads);
    }
#endif

    Option opt = _opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }

    {
        qk_softmax = ncnn::create_layer_cpu(ncnn::LayerType::Softmax);
        if (!qk_softmax)
        {
            destroy_pipeline(opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(0, -1); // axis
        pd.set(1, 1);
        int ret = qk_softmax->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        ret = qk_softmax->load_model(ModelBinFromMatArray(0));
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        ret = qk_softmax->create_pipeline(opt);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
    }

    // Q * K^T
    if (scale != 0.f)
    {
        qk_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!qk_gemm)
        {
            destroy_pipeline(opt);
            return -100;
        }
        ncnn::ParamDict pd;

        pd.set(0, scale);               // alpha
        pd.set(1, 1.f / scale);         // beta
        pd.set(2, 0);                   // transA (Q: Seq x Embed)
        pd.set(3, 1);                   // transB (K: Seq x Embed -> K^T: Embed x Seq) => Q * K^T
        pd.set(4, 0);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, attn_mask ? 0 : 1);   // constantC (if mask exists, use it)
        pd.set(7, 0);                   // M
        pd.set(8, 0);                   // N
        pd.set(9, 0);                   // K
        pd.set(10, attn_mask ? 3 : -1); // constant_broadcast_type_C (MxN)
        pd.set(11, 0);                  // output_N1M
        pd.set(12, 1);                  // output_elempack
        pd.set(13, 1);                  // output_elemtype = fp32
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        int ret = qk_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        ret = qk_gemm->load_model(ModelBinFromMatArray(0));
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        Option opt1 = opt;
        opt1.num_threads = 1;
        ret = qk_gemm->create_pipeline(opt1);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
    }

    // Attn * V
    {
        qkv_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!qkv_gemm)
        {
            destroy_pipeline(opt);
            return -100;
        }
        ncnn::ParamDict pd;
        pd.set(0, 1.f); // alpha
        pd.set(1, 1.f); // beta
        pd.set(2, 0);   // transA (Attn: Seq x Seq)
        pd.set(3, 0);   // transB (V: Seq x Embed) => Attn * V
        pd.set(4, 0);   // constantA
        pd.set(5, 0);   // constantB
        pd.set(6, 1);   // constantC (None)
        pd.set(7, 0);   // M
        pd.set(8, 0);   // N
        pd.set(9, 0);   // K
        pd.set(10, -1); // constant_broadcast_type_C
        pd.set(11, 0);  // output_N1M
        pd.set(12, 1);  // output_elempack
        pd.set(13, 1);  // output_elemtype = fp32
        pd.set(14, 0);  // output_transpose
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        int ret = qkv_gemm->load_param(pd);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        ret = qkv_gemm->load_model(ModelBinFromMatArray(0));
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
        Option opt1 = opt;
        opt1.num_threads = 1;
        ret = qkv_gemm->create_pipeline(opt1);
        if (ret != 0)
        {
            destroy_pipeline(opt);
            return ret;
        }
    }

    return 0;
}

int SDPA_riscv::destroy_pipeline(const Option& _opt)
{
    Option opt = _opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }

    if (qk_softmax)
    {
        qk_softmax->destroy_pipeline(opt);
        delete qk_softmax;
        qk_softmax = 0;
    }

    if (qk_gemm)
    {
        qk_gemm->destroy_pipeline(opt);
        delete qk_gemm;
        qk_gemm = 0;
    }

    if (qkv_gemm)
    {
        qkv_gemm->destroy_pipeline(opt);
        delete qkv_gemm;
        qkv_gemm = 0;
    }

    return 0;
}

#if __riscv_vector
/* Worst-case e32m8 VLMAX across the caller's affinity mask (the OMP team may
 * contain workers bound to either cluster): A100 vlenb 128 -> 256 lanes,
 * X100 vlenb 32 -> 64 lanes. Non-Linux or a failed query: the caller's own
 * VLMAX (the dispatching thread), which is all we can know there. */
static int sdpa_decode_vlmax_cap()
{
    int cap = (int)__riscv_vsetvlmax_e32m8();
#if defined __linux__
    /* Per-mask cache: this runs once per layer per decode token, and the walk
     * costs one sched_setaffinity syscall per allowed CPU. Masks are stable
     * within a pipeline run; recompute only when the mask changes. */
    static cpu_set_t cached_mask;
    static int cached_cap = -1;
    static ncnn::Mutex lock;
    ncnn::MutexLockGuard g(lock);
    cpu_set_t cs;
    if (sched_getaffinity(0, sizeof(cs), &cs) == 0)
    {
        if (cached_cap != -1 && memcmp(&cs, &cached_mask, sizeof(cs)) == 0)
            return cached_cap;
        const cpu_set_t orig = cs;
        for (int c = 0; c < CPU_SETSIZE; c++)
        {
            if (!CPU_ISSET(c, &cs))
                continue;
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(c, &one);
            if (sched_setaffinity(0, sizeof(one), &one) != 0)
                continue;
            unsigned long vlenb = 0;
            __asm__ volatile("csrr %0, vlenb"
                             : "=r"(vlenb));
            const int c2 = (int)(vlenb * 2); /* vlenb bytes * 8 / 32-bit lanes * m8 */
            if (c2 < cap)
                cap = c2;
        }
        sched_setaffinity(0, sizeof(orig), &orig);
        cached_mask = cs;
        cached_cap = cap;
    }
#endif
    return cap;
}
#endif // __riscv_vector

int SDPA_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& _opt) const
{
#if NCNN_BATCH
    if (kv_cache && bottom_blobs[0].n > 1)
        return -1;
#endif // NCNN_BATCH

    // ---- Fast paths added by this repo (fall through to the upstream GEMM implementation below when not taken) ----
    {
        const Mat& q0 = bottom_blobs[0];
        const char* denv = getenv("NCNN_SDPA_DECODE");
        const bool decode_enabled = !(denv && denv[0] == '0' && denv[1] == '\0');
        // Decode (M=1) RVV fast path: fp32 + kv_cache + attn_mask + single-token query
        if (decode_enabled && kv_cache && attn_mask && !int8_scale_term
                && q0.elembits() == 32 && q0.elempack == 1 && q0.h == 1
#if __riscv_vector
                /* Multi-chunk heads cannot match the fallback accumulation order:
                                                                                                                                                                         * the composed GEMM accumulates k sequentially per key (the
                                                                                                                                                                         * gemm_transB_packed_tile scalar kk loop), while any lane-grouped
                                                                                                                                                                         * chunk reduction reorders the dot product for extreme values
                                                                                                                                                                         * (both directions of the divergence were demonstrated in review).
                                                                                                                                                                         * Gate the RVV decode path to heads that fit one e32m8 vector. */
                /* Gate on the WORST worker's e32m8 VLMAX, not the caller's: an OMP
                         * head iteration may run on an X100 (vlenb 32 -> VLMAX 64) even when
                         * the dispatching thread is on an A100 (VLMAX 256), and multi-chunk
                         * reduction there reorders the dot product for E in 129..256.
                         * Walk the caller's affinity mask once per forward (per-CPU vlenb). */
                && q0.w <= sdpa_decode_vlmax_cap()
#endif
           )
        {
            /* The fast path reads the mask per key via mptr[j] (j < dst_seqlen):
             * a scalar or w==1 broadcast mask holds only one float and would be
             * read out of bounds. The GEMM composed path recognizes these
             * broadcast shapes (broadcast_type_C); fall back to it when the
             * width is insufficient. */
            const Mat& m0 = bottom_blobs[3];
            const int dst_sl = bottom_blobs[1].h + bottom_blobs[4].h; // cur + past
            if (m0.w >= dst_sl)
            {
                const int ret = forward_rvv_decode(bottom_blobs, top_blobs, _opt);
                if (ret == 0)
                    return 0;
                if (ret == -100)
                    return -100;
                // Other return values: fall back to the upstream implementation
            }
        }
    }
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    {
        const Mat& q0 = bottom_blobs[0];
        // IME2 flash prefill: takes over only for fp32 / elempack=1 / sufficiently long sequences.
        // On short sequences the fixed overhead of Q/K/V packing and tiled launch
        // outweighs the GEMM gain: measured at pp=128, enabling IME2 flash is
        // actually ~30% slower than the upstream GEMM path (33.5 -> 22.9 tok/s),
        // while at pp=2048 it is 1~2x faster. So take over only when L is large
        // enough, otherwise fall back to the upstream path.
        // The threshold can be overridden with NCNN_SDPA_IME2_MINL (for re-testing).
        int ime2_minl = 1024;
        if (const char* ml = getenv("NCNN_SDPA_IME2_MINL"))
        {
            const int v = atoi(ml);
            if (v >= 0)
                ime2_minl = v;
        }
        if (use_ime2_sdpa && kv_cache && attn_mask && !int8_scale_term
                && q0.elembits() == 32 && q0.elempack == 1 && q0.h >= ime2_minl)
        {
            /* mask shape contract: both IME2 prefill paths (flash and the old
             * row-wise one) index the mask as full [dst_seqlen, src_seqlen];
             * the upstream GEMM composed path below has real broadcast support
             * for C (broadcast_type_C runtime detection: scalar/M/N/MxN), where
             * broadcast masks such as 1xN key-padding get their correct
             * semantics. Do not take over on incomplete shapes; let the GEMM
             * path handle them.
             * Note: bottom_blobs[3]/[4] may only be accessed after confirming
             * kv_cache && attn_mask — calls without mask/cache have just 3~4
             * input blobs. */
            const Mat& m0 = bottom_blobs[3];
            const int dst_sl = bottom_blobs[1].h + bottom_blobs[4].h; // cur + past
            const bool mask_full = m0.dims >= 2 && m0.w >= dst_sl && m0.h >= q0.h;
            if (mask_full)
            {
                const int ret = forward_ime2_prefill(bottom_blobs, top_blobs, _opt);
                /* -99 = the forward team does not meet the IME2 conditions right now; fall back to the upstream GEMM path */
                if (ret != -99)
                    return ret;
            }
        }
    }
#endif
    // ---- End of fast paths; the upstream GEMM implementation follows ----

    Option opt = _opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }

    const Mat& query = bottom_blobs[0];
    const Mat& cur_key = bottom_blobs[1];
    const Mat& cur_value = bottom_blobs[2];
    const Mat& attn_mask_blob = attn_mask ? bottom_blobs[3] : Mat();
    const Mat& past_key = kv_cache ? bottom_blobs[attn_mask ? 4 : 3] : Mat();
    const Mat& past_value = kv_cache ? bottom_blobs[attn_mask ? 5 : 4] : Mat();

    const int embed_dim = query.w;
    const int src_seqlen = query.h;
    const int num_heads = query.c;
    const int cur_seqlen = cur_key.h;
    const int num_group = cur_key.c;
    const int out_embed_dim = cur_value.w;
    const int past_seqlen = kv_cache ? past_key.h : 0;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    const size_t elemsize = query.elemsize;

    Mat key;
    Mat value;
    if (kv_cache)
    {
        Mat& cached_key = top_blobs[1];
        Mat& cached_value = top_blobs[2];

        int retk = create_or_grow_kvcache(past_key, cached_key, dst_seqlen, num_group, embed_dim, elemsize, cur_key.elempack, opt);
        if (retk != 0)
            return retk;

        int retv = create_or_grow_kvcache(past_value, cached_value, dst_seqlen, num_group, out_embed_dim, elemsize, cur_value.elempack, opt);
        if (retv != 0)
            return retv;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_group; q++)
        {
            Mat key_head = cached_key.channel(q);
            Mat value_head = cached_value.channel(q);
            memcpy(key_head.row(past_seqlen), cur_key.channel(q), (size_t)embed_dim * cur_seqlen * elemsize);
            memcpy(value_head.row(past_seqlen), cur_value.channel(q), (size_t)out_embed_dim * cur_seqlen * elemsize);
        }

        key = cached_key;
        value = cached_value;
    }
    else
    {
        key = cur_key;
        value = cur_value;
    }

    const int num_heads_per_group = num_heads / num_group;

    Mat qk_cross(dst_seqlen, src_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (qk_cross.empty())
        return -100;

    std::vector<int> retqks(num_heads);

    // dynamic scale
    Layer* _qk_gemm = qk_gemm;
    if (scale == 0.f)
    {
        float _scale = 1.f / sqrt(embed_dim);

        _qk_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!_qk_gemm)
            return -100;
        ncnn::ParamDict pd;

        pd.set(0, _scale);              // alpha
        pd.set(1, 1.f / _scale);        // beta
        pd.set(2, 0);                   // transA (Q: Seq x Embed)
        pd.set(3, 1);                   // transB (K: Seq x Embed -> K^T: Embed x Seq) => Q * K^T
        pd.set(4, 0);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, attn_mask ? 0 : 1);   // constantC (if mask exists, use it)
        pd.set(7, 0);                   // M
        pd.set(8, 0);                   // N
        pd.set(9, 0);                   // K
        pd.set(10, attn_mask ? 3 : -1); // constant_broadcast_type_C (MxN)
        pd.set(11, 0);                  // output_N1M
        pd.set(12, 1);                  // output_elempack
        pd.set(13, 1);                  // output_elemtype = fp32
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        Option opt1 = opt;
        opt1.num_threads = 1;

        int ret = _qk_gemm->load_param(pd);
        if (ret == 0)
            ret = _qk_gemm->load_model(ModelBinFromMatArray(0));
        if (ret == 0)
            ret = _qk_gemm->create_pipeline(opt1);
        if (ret != 0)
        {
            _qk_gemm->destroy_pipeline(opt1);
            delete _qk_gemm;
            return ret;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        // 1. Q * K^T
        std::vector<Mat> qk_bottom_blobs;
        qk_bottom_blobs.push_back(query.channel(i));                     // Q: [Seq, Embed]
        qk_bottom_blobs.push_back(key.channel(i / num_heads_per_group)); // K: [DstSeq, Embed]

        if (attn_mask)
        {
            // Ensure mask is 2D for Gemm auto-broadcast detection
            Mat maskm = attn_mask_blob;
            if (maskm.dims == 3)
            {
                // If c > 1, pick i-th head mask. If c == 1, pick 0-th (broadcast)
                maskm = maskm.channel(maskm.c > 1 ? i : 0);
            }
            qk_bottom_blobs.push_back(maskm);
        }

        std::vector<Mat> qk_top_blobs(1);
        qk_top_blobs[0] = qk_cross.channel(i);

        Option opt1 = opt;
        opt1.num_threads = 1;
        opt1.blob_allocator = qk_cross.allocator;
        retqks[i] = _qk_gemm->forward(qk_bottom_blobs, qk_top_blobs, opt1);
    }

    if (scale == 0.f)
    {
        Option opt1 = opt;
        opt1.num_threads = 1;
        _qk_gemm->destroy_pipeline(opt1);

        delete _qk_gemm;
        _qk_gemm = 0;
    }

    for (int i = 0; i < num_heads; i++)
    {
        if (retqks[i] != 0)
            return retqks[i];
    }

    // 2. Softmax
    int retqk = qk_softmax->forward_inplace(qk_cross, opt);
    if (retqk != 0)
        return retqk;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // 3. Attn * V
    std::vector<int> retqkvs(num_heads);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        std::vector<Mat> qkv_bottom_blobs(2);
        qkv_bottom_blobs[0] = qk_cross.channel(i);                    // Attn: [DstSeq, Seq]
        qkv_bottom_blobs[1] = value.channel(i / num_heads_per_group); // V: [DstSeq, OutEmbed]

        std::vector<Mat> qkv_top_blobs(1);
        qkv_top_blobs[0] = top_blob.channel(i); // Output

        Option opt1 = opt;
        opt1.num_threads = 1;
        retqkvs[i] = qkv_gemm->forward(qkv_bottom_blobs, qkv_top_blobs, opt1);
    }

    for (int i = 0; i < num_heads; i++)
    {
        if (retqkvs[i] != 0)
            return retqkvs[i];
    }

    return 0;
}

} // namespace ncnn
