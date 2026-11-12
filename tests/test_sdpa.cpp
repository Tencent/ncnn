// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_sdpa(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int mask_type, float scale = 0.f, int flag = 0)
{
    const int src_seqlen = q.h;
    const int dst_seqlen = k.h;

    ncnn::ParamDict pd;
    pd.set(5, mask_type != 0);
    pd.set(6, scale);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (mask_type)
    {
        ncnn::Mat mask;
        if (mask_type == 2)
            mask = RandomMat(dst_seqlen, src_seqlen, 1);
        else if (mask_type == 3)
            mask = RandomMat(dst_seqlen, src_seqlen, q.c);
        else
            mask = RandomMat(dst_seqlen, src_seqlen);

        if (mask_type == 4)
        {
            const int masked_seqlen = dst_seqlen / 2;
            for (int i = 0; i < src_seqlen; i++)
            {
                float* mptr = mask.row(i);
                for (int j = 0; j < masked_seqlen; j++)
                    mptr[j] = -10000.f;
            }
        }

        as.push_back(mask);
    }

    float epsilon = 0.001;

    int ret = test_layer("SDPA", pd, weights, as, 1, epsilon, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) mask_type=%d scale=%f\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, mask_type, scale);
    }

    return ret;
}

static int test_sdpa_0()
{
    return 0
           || test_sdpa(RandomMat(64, 1, 16), RandomMat(64, 521, 1), RandomMat(48, 521, 1), 0)
           || test_sdpa(RandomMat(63, 1, 31), RandomMat(63, 37, 1), RandomMat(47, 37, 1), 3, 0.f, TEST_LAYER_ENABLE_THREADING)
           || test_sdpa(RandomMat(65, 1, 4), RandomMat(65, 37, 4), RandomMat(33, 37, 4), 1)
           || test_sdpa(RandomMat(80, 1, 8), RandomMat(80, 521, 2), RandomMat(96, 521, 2), 1, -0.4f)
           || test_sdpa(RandomMat(63, 31, 15), RandomMat(63, 31, 3), RandomMat(47, 31, 3), 2, 0.2f)
           || test_sdpa(RandomMat(27, 5, 15), RandomMat(27, 37, 3), RandomMat(23, 37, 3), 3, 0.f, TEST_LAYER_ENABLE_THREADING)
           || test_sdpa(RandomMat(128, 17, 4), RandomMat(128, 127, 4), RandomMat(192, 127, 4), 4)
           || test_sdpa(RandomMat(32, 66, 8), RandomMat(32, 66, 8), RandomMat(20, 66, 8), 0)
           || test_sdpa(RandomMat(26, 64, 8), RandomMat(26, 61, 8), RandomMat(18, 61, 8), 1)
           || test_sdpa(RandomMat(64, 128, 12), RandomMat(64, 128, 2), RandomMat(64, 128, 2), 0)
           || test_sdpa(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, -0.4f);
}

#if NCNN_INT8
static int test_sdpa_int8(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, float scale = 0.f)
{
    const int src_seqlen = q.h;
    const int dst_seqlen = k.h;

    ncnn::ParamDict pd;
    pd.set(5, attn_mask);
    pd.set(6, scale);
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(RandomMat(dst_seqlen, src_seqlen));
    }

    float epsilon = 0.01;

    int ret = test_layer("SDPA", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_int8 failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d scale=%f\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, scale);
    }

    return ret;
}

static int test_sdpa_1()
{
    return 0
           || test_sdpa_int8(RandomMat(32, 66, 8), RandomMat(32, 66, 8), RandomMat(20, 66, 8), 0)
           || test_sdpa_int8(RandomMat(26, 64, 8), RandomMat(26, 61, 8), RandomMat(18, 61, 8), 1)
           || test_sdpa_int8(RandomMat(40, 62, 7), RandomMat(40, 61, 7), RandomMat(24, 61, 7), 0)
           || test_sdpa_int8(RandomMat(24, 22, 6), RandomMat(24, 19, 6), RandomMat(16, 19, 6), 1)
           || test_sdpa_int8(RandomMat(64, 128, 12), RandomMat(64, 128, 2), RandomMat(64, 128, 2), 0)
           || test_sdpa_int8(RandomMat(48, 122, 12), RandomMat(64, 127, 2), RandomMat(64, 127, 2), 1)
           || test_sdpa_int8(RandomMat(44, 128, 4), RandomMat(44, 123, 4), RandomMat(55, 123, 4), 0, 1.f)
           || test_sdpa_int8(RandomMat(12, 127, 4), RandomMat(12, 127, 4), RandomMat(55, 127, 4), 1, 1.f)
           || test_sdpa_int8(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 0.1f)
           || test_sdpa_int8(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, -0.4f);
}
#endif

#if NCNN_VULKAN
// flash attention reduces each row of the score matrix on one subgroup when the device supports
// subgroup ops and they are enabled, the shader picks between that and its barrier tree itself,
// so compare both of them against the generic path, which fp32 storage takes
static int test_sdpa_gpu_flash_attention(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, bool with_mask)
{
    const int src_seqlen = q.h;
    const int dst_seqlen = k.h;

    ncnn::ParamDict pd;
    pd.set(5, with_mask ? 1 : 0);
    pd.set(6, 0.f);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (with_mask)
    {
        as.push_back(RandomMat(dst_seqlen, src_seqlen));
    }

    const ncnn::GpuInfo& info = ncnn::get_gpu_device()->info;

    // fp16 and bf16 storage take the flash attention path, the fp32 run is the reference
    const bool fp16[2] = {true, false};
    const bool bf16[2] = {false, true};
    const char* names[2] = {"fp16ps", "bf16ps"};

    ncnn::Option opt_ref;
    opt_ref.num_threads = 1;
    opt_ref.use_vulkan_compute = true;
    opt_ref.use_fp16_packed = false;
    opt_ref.use_fp16_storage = false;
    opt_ref.use_fp16_arithmetic = false;
    opt_ref.use_bf16_packed = false;
    opt_ref.use_bf16_storage = false;

    std::vector<ncnn::Mat> d0;
    int ret = test_layer_gpu(ncnn::layer_to_index("SDPA"), pd, weights, opt_ref, as, 1, d0, std::vector<ncnn::Mat>(), 0);
    if (ret == 233)
        return 0;
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_gpu_flash_attention generic failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) mask=%d ret=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, with_mask, ret);
        return -1;
    }

    for (int i = 0; i < 2; i++)
    {
        if (fp16[i] && !info.support_fp16_storage())
            continue;
        if (bf16[i] && !info.support_bf16_storage())
            continue;

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_vulkan_compute = true;
        opt.use_fp16_packed = fp16[i];
        opt.use_fp16_storage = fp16[i];
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = bf16[i];
        opt.use_bf16_storage = bf16[i];

        std::vector<ncnn::Mat> d1;
        ret = test_layer_gpu(ncnn::layer_to_index("SDPA"), pd, weights, opt, as, 1, d1, std::vector<ncnn::Mat>(), 0);
        if (ret == 233)
            continue;
        if (ret != 0)
        {
            fprintf(stderr, "test_sdpa_gpu_flash_attention failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) mask=%d %s ret=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, with_mask, names[i], ret);
            return -1;
        }

        if (CompareMat(d0, d1, 0.01f) != 0)
        {
            fprintf(stderr, "test_sdpa_gpu_flash_attention mismatch q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) mask=%d %s\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, with_mask, names[i]);
            return -1;
        }

        // with subgroup ops switched off the row reduction falls back to the barrier tree
        opt.use_subgroup_ops = false;
        std::vector<ncnn::Mat> d2;
        ret = test_layer_gpu(ncnn::layer_to_index("SDPA"), pd, weights, opt, as, 1, d2, std::vector<ncnn::Mat>(), 0);
        if (ret != 0)
        {
            fprintf(stderr, "test_sdpa_gpu_flash_attention barrier tree failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) mask=%d %s ret=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, with_mask, names[i], ret);
            return -1;
        }

        if (CompareMat(d0, d2, 0.01f) != 0 || CompareMat(d1, d2, 0.01f) != 0)
        {
            fprintf(stderr, "test_sdpa_gpu_flash_attention barrier tree mismatch q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) mask=%d %s\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, with_mask, names[i]);
            return -1;
        }
    }

    return 0;
}

static int test_sdpa_subgroup()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    // several query rows per workgroup, and a key length that is not a multiple of the tile
    return 0
           || test_sdpa_gpu_flash_attention(RandomMat(64, 8, 4), RandomMat(64, 64, 2), RandomMat(64, 64, 2), false)
           || test_sdpa_gpu_flash_attention(RandomMat(128, 5, 8), RandomMat(128, 37, 8), RandomMat(128, 37, 8), false)
           || test_sdpa_gpu_flash_attention(RandomMat(64, 4, 4), RandomMat(64, 32, 4), RandomMat(48, 32, 4), true);
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return test_sdpa_0() || test_sdpa_1()
#if NCNN_VULKAN
           || test_sdpa_subgroup()
#endif
           ;
#else
    return test_sdpa_0()
#if NCNN_VULKAN
           || test_sdpa_subgroup()
#endif
           ;
#endif
}
