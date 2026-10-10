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

int SDPA_riscv::create_pipeline(const Option& _opt)
{
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    // IME2 探测在 zfh 变体 TU 里做（那里有 smt.vfwmadot 的汇编）
    use_ime2_sdpa = ime2_available();
    // 调试开关：NCNN_SDPA_IME2=0 强制关闭（定位问题用）
    const char* ime2env = getenv("NCNN_SDPA_IME2");
    if (ime2env && ime2env[0] == '0' && ime2env[1] == '\0')
        use_ime2_sdpa = 0;
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
            const int ret = forward_rvv_decode(bottom_blobs, top_blobs, _opt);
            if (ret == 0)
                return 0;
            if (ret == -100)
                return -100;
            // 其它返回值：回落上游实现
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
            return forward_ime2_prefill(bottom_blobs, top_blobs, _opt);
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
