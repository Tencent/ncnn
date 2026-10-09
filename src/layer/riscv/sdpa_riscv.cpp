// SpacemiT K3 IME2 SDPA acceleration
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_riscv.h"
#include <cstdlib>
#include <cstdio>

#include <stdlib.h>

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
#include <math.h>
#endif

#if __riscv_v
#include <riscv_vector.h>
#endif

#include <float.h>
#include <math.h>
#include <string.h>

namespace ncnn {

#if __riscv_v
// ---------------- RVV 解码辅助（M=1 注意力，f32） ----------------
// 全部显式 vsetvl（VLEN=1024 下 e32m8 的 VL 上限 256，写死 zero 会越界）。

#if __riscv_vector
// 向量化 exp（与 swish_riscv.cpp 同一套实现）：exp(x) = 2^(x*log2e) = 2^n * 2^f
// 逐元素运算，结果与 VLEN 无关 —— 满足跨簇逐字节一致这一硬门槛。
// 解码 softmax 原本用标量 expf（实测 44.7 ns/元素），507 token 时每层 8112 次 ≈ 363 µs，
// 是长上下文解码的主要成本（每 token 约 10 ms）。这里换成向量版。
static inline vfloat32m1_t sdpa_vexp_f32m1(vfloat32m1_t x, size_t vl)
{
    const vfloat32m1_t vlog2e = __riscv_vfmv_v_f_f32m1(1.4426950408889634f, vl);
    const vfloat32m1_t vhalf = __riscv_vfmv_v_f_f32m1(0.5f, vl);
    vfloat32m1_t t = __riscv_vfmul_vv_f32m1(x, vlog2e, vl);
    vfloat32m1_t nf = __riscv_vfadd_vv_f32m1(t, vhalf, vl);
    nf = __riscv_vfmin_vf_f32m1(nf, 127.f, vl);
    nf = __riscv_vfmax_vf_f32m1(nf, -126.f, vl);
    vint32m1_t ni = __riscv_vfcvt_x_f_v_i32m1_rm(nf, __RISCV_FRM_RDN, vl);
    vfloat32m1_t n = __riscv_vfcvt_f_x_v_f32m1(ni, vl);
    vfloat32m1_t f = __riscv_vfsub_vv_f32m1(t, n, vl);
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(1.3392112e-4f, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(1.3392112e-3f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(9.6181291e-3f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(5.5504109e-2f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(2.4022651e-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(6.9314718e-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);
    vint32m1_t bias = __riscv_vsll_vx_i32m1(ni, 23, vl);
    vfloat32m1_t scale = __riscv_vreinterpret_v_i32m1_f32m1(__riscv_vadd_vx_i32m1(bias, 0x3f800000, vl));
    return __riscv_vfmul_vv_f32m1(p, scale, vl);
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
    int k = 0;
    size_t vl = __riscv_vsetvl_e32m8(n);
    size_t last_vl = vl;
    vfloat32m8_t acc = __riscv_vfmul_vv_f32m8(__riscv_vle32_v_f32m8(a, vl), __riscv_vle32_v_f32m8(b, vl), vl);
    k += (int)vl;
    while (k < n)
    {
        vl = __riscv_vsetvl_e32m8(n - k);
        acc = __riscv_vfmacc_vv_f32m8(acc, __riscv_vle32_v_f32m8(a + k, vl), __riscv_vle32_v_f32m8(b + k, vl), vl);
        k += (int)vl;
        last_vl = vl;
    }
    // 归约的 vl 必须等于**最后一次迭代**的 vl：若用 vsetvl(n) 而尾块不满，
    // 尾块之前的元素会被重复累加（VLMAX 随 VLEN 变化，小 VLEN 簇上必然触发）。
    vfloat32m1_t s = __riscv_vfredusum_vs_f32m8_f32m1(acc, __riscv_vfmv_v_f_f32m1(0.f, 1), last_vl);
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

SDPA_riscv::SDPA_riscv()
{
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    use_ime2_sdpa = 0;
#endif
}

int SDPA_riscv::create_pipeline(const Option& opt)
{
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    // 探测在 zfh 变体 TU 里做（那里有 smt.vfwmadot 的汇编）
    use_ime2_sdpa = ime2_available();
    // 调试开关：NCNN_SDPA_IME2=0 可强制关闭（定位问题用）
    const char* env = getenv("NCNN_SDPA_IME2");
    if (env && env[0] == '0' && env[1] == '\0')
        use_ime2_sdpa = 0;
#endif

    return SDPA::create_pipeline(opt);
}

// 解码（M=1）注意力：QK^T 逐位置向量点积 + softmax + 流式 P·V。
// 与朴素路径逐项同序（求和顺序不同，偏差在 fp32 舍入量级），语义完全一致。
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
    float* score_slab = (float*)opt.workspace_allocator->fastMalloc((size_t)nthreads * dst_seqlen * sizeof(float));
    if (!score_slab)
        return -1;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty())
    {
        opt.workspace_allocator->fastFree(score_slab);
        return -100;
    }

    // ---- kv cache 追加（与朴素路径逐字节相同的语义） ----
    Mat& cached_key = top_blobs[1];
    Mat& cached_value = top_blobs[2];

    int retk = create_or_grow_kvcache(past_key, cached_key, dst_seqlen, num_group, embed_dim, cur_key.elemsize, cur_key.elempack, opt);
    if (retk != 0)
    {
        opt.workspace_allocator->fastFree(score_slab);
        return retk;
    }
    int retv = create_or_grow_kvcache(past_value, cached_value, dst_seqlen, num_group, out_embed_dim, cur_value.elemsize, cur_value.elempack, opt);
    if (retv != 0)
    {
        opt.workspace_allocator->fastFree(score_slab);
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
            int j = 0;
            vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
            while (j < dst_seqlen)
            {
                const size_t vl = __riscv_vsetvl_e32m1(dst_seqlen - j);
                vfloat32m1_t v = __riscv_vle32_v_f32m1(scores + j, vl);
                v = sdpa_vexp_f32m1(__riscv_vfsub_vf_f32m1(v, mx, vl), vl);
                __riscv_vse32_v_f32m1(scores + j, v, vl);
                vsum = __riscv_vfadd_vv_f32m1(vsum, v, vl);
                j += (int)vl;
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vsum, __riscv_vfmv_v_f_f32m1(0.f, 1), __riscv_vsetvlmax_e32m1()));
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

    opt.workspace_allocator->fastFree(score_slab);
    return 0;
#endif // __riscv_v
}

int SDPA_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // 解码（M=1）RVV 快速路径：fp32 + kv_cache + attn_mask + 单 token 查询
    {
        const Mat& query = bottom_blobs[0];
        const char* denv = getenv("NCNN_SDPA_DECODE");
        const bool decode_enabled = !(denv && denv[0] == '0' && denv[1] == '\0');
        if (decode_enabled && kv_cache && attn_mask && !int8_scale_term && query.elembits() == 32 && query.elempack == 1 && query.h == 1)
        {
            const int ret = forward_rvv_decode(bottom_blobs, top_blobs, opt);
            if (ret == 0)
                return 0;
            // ret != 0：回退朴素实现（-100 分配失败则直接报错）
            if (ret == -100)
                return -100;
        }
    }

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    if (use_ime2_sdpa && kv_cache && attn_mask && !int8_scale_term)
    {
        const Mat& query = bottom_blobs[0];
        // 只接管 fp32/elempack=1 的预填充（q 序列长度 > 1）；解码(L=1)走上面的 RVV 路径
        if (query.elembits() == 32 && query.elempack == 1 && query.h > 1)
            return forward_ime2_prefill(bottom_blobs, top_blobs, opt);
    }
#endif

    return SDPA::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
