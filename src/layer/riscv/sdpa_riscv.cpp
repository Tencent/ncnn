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
// ---------------- RVV 解码辅助（M=1 注意力，f32） ----------------
// 全部显式 vsetvl（VLEN=1024 下 e32m8 的 VL 上限 256，写死 zero 会越界）。

#if __riscv_vector
// 向量化 exp（与 swish_riscv.cpp 同一套实现）：exp(x) = 2^(x*log2e) = 2^n * 2^f
// 逐元素运算，结果与 VLEN 无关。
// 注意跨簇逐字节一致的边界：exp 与本层 softmax 求和（固定 8 lane + vfredosum）
// 是 VLEN 无关的；但 rvv_dot_f32（QK）与 PV 累加仍是按 lane 分组（e32m8 宽度
// 随 VLEN 变化），所以整条解码路径只在**同一簇**上保证逐字节可复现——K3 上
// 进程注册后被限制在 A100 簇内，这一前提成立；与 ncnn 全局一致（打包 tile
// 的求和顺序本来就随 VLEN 变，上游不承诺跨 VLEN 位一致）。
// 解码 softmax 原本用标量 expf（实测 44.7 ns/元素），507 token 时每层 8112 次 ≈ 363 µs，
// 是长上下文解码的主要成本（每 token 约 10 ms）。这里换成向量版。
static inline vfloat32m1_t sdpa_vexp_f32m1(vfloat32m1_t x, size_t vl)
{
    /* 与上游 Softmax_riscv 使用的 exp_ps（rvv_mathfun.h）逐位一致的构造：
     * 同一 cephes 多项式、同一钳位常数、同一 +0.5/floor 取整、同一 ldexp
     * 位技巧。解码快路径的输出与组合回退路径逐位相同；低于 exp_lo 的值与
     * 上游一样冲刷为 0（NaN 的钳位语义也与上游一致：vfmin/vfmax 遇单 NaN
     * 取数值操作数）。 */
    /* 常数名加 sdpa_ 前缀：rvv_mathfun.h 里这些是宏，同名局部声明会冲突。 */
    const float sdpa_exp_hi = 88.3762626647949f;
    const float sdpa_exp_lo = -88.3762626647949f; /* 上游 exp_ps 的实际钳位下界 */
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

    /* floorf：转整再转回，比 fx 大则减 1（与上游逐字相同的 floor 模拟） */
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
    /* 与上游一致：分开的乘加（两次舍入），不是 FMA——否则末位差 1 ulp。 */
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
    /* 与上游 Softmax_riscv 使用的 exp_ps（rvv_mathfun.h）逐位一致的构造：
     * 同一 cephes 多项式、同一钳位常数、同一 +0.5/floor 取整、同一 ldexp
     * 位技巧。解码快路径的输出与组合回退路径逐位相同；低于 exp_lo 的值与
     * 上游一样冲刷为 0（NaN 的钳位语义也与上游一致：vfmin/vfmax 遇单 NaN
     * 取数值操作数）。 */
    /* 常数名加 sdpa_ 前缀：rvv_mathfun.h 里这些是宏，同名局部声明会冲突。 */
    const float sdpa_exp_hi = 88.3762626647949f;
    const float sdpa_exp_lo = -88.3762626647949f; /* 上游 exp_ps 的实际钳位下界 */
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

    /* floorf：转整再转回，比 fx 大则减 1（与上游逐字相同的 floor 模拟） */
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
    /* 与上游一致：分开的乘加（两次舍入），不是 FMA——否则末位差 1 ulp。 */
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
// 解码注意力分段计时的本地时钟（不依赖 IME2 侧 profiler，保证所有变体都能编译）
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
    float total = 0.f;
    int k = 0;
    while (k < n)
    {
        const size_t vl = __riscv_vsetvl_e32m8(n - k);
        vfloat32m8_t p = __riscv_vfmul_vv_f32m8(__riscv_vle32_v_f32m8(a + k, vl), __riscv_vle32_v_f32m8(b + k, vl), vl);
        /* vfredusum only reduces the first vl lanes, so the tail is not read */
        vfloat32m1_t s = __riscv_vfredusum_vs_f32m8_f32m1(p, __riscv_vfmv_v_f_f32m1(0.f, 1), vl);
        total += __riscv_vfmv_f_s_f32m1_f32(s);
        k += (int)vl;
    }
    return total;
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

    // 每线程一份 scores scratch。必须在**改动 KV cache 之前**分配好：
    // 一旦追加过 cache 就不能再回退朴素路径（会重复追加）。
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

    // ---- kv cache 追加（与朴素路径逐字节相同的语义） ----
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

    // ---- 逐头注意力 ----
    #pragma omp parallel for num_threads(opt.num_threads)
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

        // QK^T：scores[j] = (q · k_j) * scale + mask[j]
        for (int j = 0; j < dst_seqlen; j++)
        {
            const float* kptr = key_head.row(j);
            scores[j] = rvv_dot_f32(qptr, kptr, embed_dim) * _scale + mptr[j];
        }

        const double dt1 = dprof ? sdpa_dec_now() : 0.0;

        // softmax（与朴素路径同序：max → expf → sum → 乘倒数）
        const float mx = rvv_max_f32(scores, dst_seqlen);
        float sum = 0.f;
#if __riscv_vector
        /* 向量化 exp（原来逐元素调 expf，44.7 ns/元素 → 507 token 时每层约 363 µs） */
        {
            /* 求和的分组必须与 VLEN 无关：lane 数跟随 VLEN（A100 e32m1=32 条，
             * X100=16 条），lane 分组累加 + 无序归约会让 sum 跨簇不同。
             * 固定 8 lane 分组（任何支持的 VLEN 都有 VLMAX≥8），归约用
             * vfredosum（有序，顺序语义固定）——两簇逐字节一致。exp 本身是
             * 累加结构与上游 Softmax_riscv 逐字一致（e32m8 大组 + _tu 末组 +
             * vfredusum 收尾）：小项如果都落到最大值所在的 lane 会被逐个舍掉
             * （1 + 2.5e-8 == 1），大组跨步把小项分散到各 lane 的独立累加器里
             * 才能留住聚合质量——与回退路径逐位一致的前提。 */
            vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvlmax_e32m8());
            {
                const size_t vlfull = __riscv_vsetvlmax_e32m8();
                float* ptr = scores;
                int n = dst_seqlen / (int)vlfull * (int)vlfull;
                while (n > 0)
                {
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(ptr, vlfull);
                    v = __riscv_vfsub_vf_f32m8(v, mx, vlfull);
                    /* m8 版解码 exp：与 exp_ps 相同构造的宽版（下方） */
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
                /* 与上游相同：vfredusum（无序但在同一实现上确定） */
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

        // P·V：out[e] += p_j * v_j[e]，输出行驻留寄存器。
        // 注意：必须按 e32m8 的 **VLMAX**（随 VLEN 变化：VLEN=1024 → 256，VLEN=256 → 64）
        // 判断能否一次写完，否则在 X100 这类小 VLEN 簇上只会写出前 vl 个元素，
        // 其余保持未初始化 → 输出乱码（本 bug 曾被"X100 那一腿实际未生效"的闸门掩盖）。
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

    return 0;
#endif // __riscv_v
}

int SDPA_riscv::create_pipeline(const Option& _opt)
{
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    // 调试开关：NCNN_SDPA_IME2=0 强制关闭（定位问题用）。必须先于探测：
    // 否则强制关闭也照样付一次逐核亲和性遍历 + 进程级 SIGILL 探针。
    const char* ime2env = getenv("NCNN_SDPA_IME2");
    if (ime2env && ime2env[0] == '0' && ime2env[1] == '\0')
    {
        use_ime2_sdpa = 0;
    }
    else
    {
        // IME2 探测在 zfh 变体 TU 里做（那里有 smt.vfwmadot 的汇编）
        // 用与 forward 相同的团队大小探测：forward 的并行区都带 num_threads(opt.num_threads)
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

int SDPA_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& _opt) const
{
#if NCNN_BATCH
    if (kv_cache && bottom_blobs[0].n > 1)
        return -1;
#endif // NCNN_BATCH

    // ---- 本仓库新增的快速路径（不命中时继续走下面的上游 GEMM 实现）----
    {
        const Mat& q0 = bottom_blobs[0];
        const char* denv = getenv("NCNN_SDPA_DECODE");
        const bool decode_enabled = !(denv && denv[0] == '0' && denv[1] == '\0');
        // 解码（M=1）RVV 快速路径：fp32 + kv_cache + attn_mask + 单 token 查询
        if (decode_enabled && kv_cache && attn_mask && !int8_scale_term
                && q0.elembits() == 32 && q0.elempack == 1 && q0.h == 1)
        {
            /* 快速路径按 mptr[j]（j < dst_seqlen）逐键读 mask：标量或 w==1
             * 的广播 mask 只有一个 float，会越界。GEMM 组合路径能识别这些
             * 广播形状（broadcast_type_C），宽度不够就回落到那里。 */
            const Mat& m0 = bottom_blobs[3];
            const int dst_sl = bottom_blobs[1].h + bottom_blobs[4].h; // cur + past
            if (m0.w >= dst_sl)
            {
                const int ret = forward_rvv_decode(bottom_blobs, top_blobs, _opt);
                if (ret == 0)
                    return 0;
                if (ret == -100)
                    return -100;
                // 其它返回值：回落上游实现
            }
        }
    }
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    {
        const Mat& q0 = bottom_blobs[0];
        // IME2 flash 预填充：只接管 fp32 / elempack=1 / 序列长度足够长。
        // 短序列下 Q/K/V 打包与分块启动的固定开销超过 GEMM 收益：实测 pp=128 时
        // 打开 IME2 flash 反而比上游 GEMM 路径慢约 30%（33.5 -> 22.9 tok/s），
        // 而 pp=2048 快 1~2 倍。所以只在 L 足够大时才接管，否则回落上游路径。
        // 阈值可用 NCNN_SDPA_IME2_MINL 覆盖（便于复测）。
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
            /* mask 形状契约：IME2 预填充的两条路径（flash 与旧逐行）都按完整
             * [dst_seqlen, src_seqlen] 索引 mask；而下面的上游 GEMM 组合路径对 C
             * 有真正的广播支持（broadcast_type_C 运行时检测：标量/M/N/MxN），
             * 1xN key-padding 之类的广播 mask 在那里才是正确语义。形状不完整时
             * 不接管，让 GEMM 路径处理。
             * 注意：bottom_blobs[3]/[4] 的访问必须在确认 kv_cache && attn_mask
             * 之后——无 mask/无 cache 的调用只有 3~4 个输入 blob。 */
            const Mat& m0 = bottom_blobs[3];
            const int dst_sl = bottom_blobs[1].h + bottom_blobs[4].h; // cur + past
            const bool mask_full = m0.dims >= 2 && m0.w >= dst_sl && m0.h >= q0.h;
            if (mask_full)
            {
                const int ret = forward_ime2_prefill(bottom_blobs, top_blobs, _opt);
                /* -99 = forward 团队此刻不满足 IME2 条件，回落上游 GEMM 路径 */
                if (ret != -99)
                    return ret;
            }
        }
    }
#endif
    // ---- 快速路径结束；以下为上游 GEMM 实现 ----

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
