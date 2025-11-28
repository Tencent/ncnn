// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "cpu.h"
#include "layer.h"
#include "mat.h"
#include "prng.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

static struct prng_rand_t g_prng_rand_state;

void SRAND(int seed)
{
    prng_srand(seed, &g_prng_rand_state);
}

uint64_t RAND()
{
    return prng_rand(&g_prng_rand_state);
}

float RandomFloat(float a, float b)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    float v = a + r;
    // generate denormal as zero
    if (v < 0.0001 && v > -0.0001)
        v = 0.f;
    return v;
}

int RandomInt(int a, int b)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    int diff = b - a;
    float r = random * diff;
    return a + (int)r;
}

signed char RandomS8()
{
    return (signed char)RandomInt(-127, 127);
}

void Randomize(ncnn::Mat& m, float a, float b)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        m[i] = RandomFloat(a, b);
    }
}

void RandomizeInt(ncnn::Mat& m, int a, int b)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        ((int*)m)[i] = RandomInt(a, b);
    }
}

void RandomizeS8(ncnn::Mat& m)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        ((signed char*)m)[i] = RandomS8();
    }
}

ncnn::Mat RandomMat(int w, float a, float b)
{
    ncnn::Mat m(w);
    Randomize(m, a, b);
    return m;
}

ncnn::Mat RandomMat(int w, int h, float a, float b)
{
    ncnn::Mat m(w, h);
    Randomize(m, a, b);
    return m;
}

ncnn::Mat RandomMat(int w, int h, int c, float a, float b)
{
    ncnn::Mat m(w, h, c);
    Randomize(m, a, b);
    return m;
}

ncnn::Mat RandomMat(int w, int h, int d, int c, float a, float b)
{
    ncnn::Mat m(w, h, d, c);
    Randomize(m, a, b);
    return m;
}

ncnn::Mat RandomIntMat(int w)
{
    ncnn::Mat m(w);
    RandomizeInt(m);
    return m;
}

ncnn::Mat RandomIntMat(int w, int h)
{
    ncnn::Mat m(w, h);
    RandomizeInt(m);
    return m;
}

ncnn::Mat RandomIntMat(int w, int h, int c)
{
    ncnn::Mat m(w, h, c);
    RandomizeInt(m);
    return m;
}

ncnn::Mat RandomIntMat(int w, int h, int d, int c)
{
    ncnn::Mat m(w, h, d, c);
    RandomizeInt(m);
    return m;
}

ncnn::Mat RandomS8Mat(int w)
{
    ncnn::Mat m(w, (size_t)1u);
    RandomizeS8(m);
    return m;
}

ncnn::Mat RandomS8Mat(int w, int h)
{
    ncnn::Mat m(w, h, (size_t)1u);
    RandomizeS8(m);
    return m;
}

ncnn::Mat RandomS8Mat(int w, int h, int c)
{
    ncnn::Mat m(w, h, c, (size_t)1u);
    RandomizeS8(m);
    return m;
}

ncnn::Mat RandomS8Mat(int w, int h, int d, int c)
{
    ncnn::Mat m(w, h, d, c, (size_t)1u);
    RandomizeS8(m);
    return m;
}

ncnn::Mat scales_mat(const ncnn::Mat& mat, int m, int k, int ldx)
{
    ncnn::Mat weight_scales(m);
    for (int i = 0; i < m; ++i)
    {
        float min = mat[0], _max = mat[0];
        const float* ptr = (const float*)(mat.data) + i * ldx;
        for (int j = 0; j < k; ++j)
        {
            if (min > ptr[j])
            {
                min = ptr[j];
            }
            if (_max < ptr[j])
            {
                _max = ptr[j];
            }
        }
        const float abs_min = abs(min), abs_max = abs(_max);
        weight_scales[i] = 127.f / (abs_min > abs_max ? abs_min : abs_max);
    }
    return weight_scales;
}

bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = (float)fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

int Compare(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon)
{
#define CHECK_MEMBER(m)                                                                 \
    if (a.m != b.m)                                                                     \
    {                                                                                   \
        fprintf(stderr, #m " not match    expect %d but got %d\n", (int)a.m, (int)b.m); \
        return -1;                                                                      \
    }

    CHECK_MEMBER(dims)
    CHECK_MEMBER(w)
    CHECK_MEMBER(h)
    CHECK_MEMBER(d)
    CHECK_MEMBER(c)
    CHECK_MEMBER(elemsize)
    CHECK_MEMBER(elempack)

#undef CHECK_MEMBER

    for (int q = 0; q < a.c; q++)
    {
        const ncnn::Mat ma = a.channel(q);
        const ncnn::Mat mb = b.channel(q);
        for (int z = 0; z < a.d; z++)
        {
            const ncnn::Mat da = ma.depth(z);
            const ncnn::Mat db = mb.depth(z);
            for (int i = 0; i < a.h; i++)
            {
                const float* pa = da.row(i);
                const float* pb = db.row(i);
                for (int j = 0; j < a.w; j++)
                {
                    if (!NearlyEqual(pa[j], pb[j], epsilon))
                    {
                        fprintf(stderr, "value not match  at c:%d d:%d h:%d w:%d    expect %f but got %f\n", q, z, i, j, pa[j], pb[j]);
                        return -1;
                    }
                }
            }
        }
    }

    return 0;
}

int CompareMat(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon)
{
    ncnn::Option opt;
    opt.num_threads = 1;

    if (a.elempack != 1)
    {
        ncnn::Mat a1;
        ncnn::convert_packing(a, a1, 1, opt);
        return CompareMat(a1, b, epsilon);
    }

    if (b.elempack != 1)
    {
        ncnn::Mat b1;
        ncnn::convert_packing(b, b1, 1, opt);
        return CompareMat(a, b1, epsilon);
    }

    if (a.elemsize == 2u)
    {
        ncnn::Mat a32;
        cast_float16_to_float32(a, a32, opt);
        return CompareMat(a32, b, epsilon);
    }
    if (a.elemsize == 1u)
    {
        ncnn::Mat a32;
        cast_int8_to_float32(a, a32, opt);
        return CompareMat(a32, b, epsilon);
    }

    if (b.elemsize == 2u)
    {
        ncnn::Mat b32;
        cast_float16_to_float32(b, b32, opt);
        return CompareMat(a, b32, epsilon);
    }
    if (b.elemsize == 1u)
    {
        ncnn::Mat b32;
        cast_int8_to_float32(b, b32, opt);
        return CompareMat(a, b32, epsilon);
    }

    return Compare(a, b, epsilon);
}

int CompareMat(const std::vector<ncnn::Mat>& a, const std::vector<ncnn::Mat>& b, float epsilon)
{
    if (a.size() != b.size())
    {
        fprintf(stderr, "output blob count not match %zu %zu\n", a.size(), b.size());
        return -1;
    }

    for (size_t i = 0; i < a.size(); i++)
    {
        if (CompareMat(a[i], b[i], epsilon))
        {
            fprintf(stderr, "output blob %zu not match\n", i);
            return -1;
        }
    }

    return 0;
}

static int convert_to_optimal_layout(const ncnn::Mat& a, ncnn::Mat& a4, ncnn::Mat& ax, const ncnn::Option& opt, const ncnn::Layer* op, int flag)
{
    // clang-format off
    // *INDENT-OFF*
#if NCNN_ARM82
    if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::cast_float32_to_float16(a, a4, opt);
    }
    else
#endif // NCNN_ARM82
#if NCNN_VFPV4
    if (opt.use_fp16_storage && !opt.use_bf16_storage && ncnn::cpu_support_arm_vfpv4() && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::cast_float32_to_float16(a, a4, opt);
    }
    else
#endif // NCNN_VFPV4
#if NCNN_ZFH
    if (opt.use_fp16_storage && (ncnn::cpu_support_riscv_zvfh() || (!ncnn::cpu_support_riscv_v() && ncnn::cpu_support_riscv_zfh())) && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::cast_float32_to_float16(a, a4, opt);
    }
    else
#endif // NCNN_ZFH
#if NCNN_BF16
    if (opt.use_bf16_storage && op->support_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::cast_float32_to_bfloat16(a, a4, opt);
    }
    else
#endif // NCNN_BF16
    if (opt.use_fp16_storage && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::cast_float32_to_float16(a, a4, opt);
    }
    else
    {
        a4 = a;
    }
    // *INDENT-ON*
    // clang-format on

    ax = a4;

    if (opt.use_packing_layout && op->support_packing && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_PACKING))
    {
        // resolve dst_elempack
        int dims = a4.dims;
        int elemcount = 0;
        if (dims == 1) elemcount = a4.elempack * a4.w;
        if (dims == 2) elemcount = a4.elempack * a4.h;
        if (dims == 3 || dims == 4) elemcount = a4.elempack * a4.c;

        int elembits = a4.elembits();

        int dst_elempack = 1;

        if (elembits == 32)
        {
#if NCNN_AVX512
            if (elemcount % 16 == 0 && ncnn::cpu_support_x86_avx512())
                dst_elempack = 16;
            else if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#elif NCNN_AVX
            if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#elif NCNN_RVV || NCNN_XTHEADVECTOR
            const int packn = ncnn::cpu_riscv_vlenb() / 4;
            if (elemcount % packn == 0)
                dst_elempack = packn;
#else
            if (elemcount % 4 == 0)
                dst_elempack = 4;
#endif
        }
        if (elembits == 16)
        {
#if NCNN_ARM82
            if (elemcount % 8 == 0 && ncnn::cpu_support_arm_asimdhp() && opt.use_fp16_arithmetic && op->support_fp16_storage)
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#elif NCNN_RVV || NCNN_XTHEADVECTOR
            const int packn = ncnn::cpu_riscv_vlenb() / 2;
            if (elemcount % packn == 0)
                dst_elempack = packn;
#else
            if (elemcount % 4 == 0)
                dst_elempack = 4;
#endif
        }
        if (elembits == 8)
        {
#if NCNN_RVV || NCNN_XTHEADVECTOR
            const int packn = ncnn::cpu_riscv_vlenb() / 1;
            if (elemcount % packn == 0)
                dst_elempack = packn;
#else
            if (elemcount % 8 == 0)
                dst_elempack = 8;
#endif
        }

        if (flag & TEST_LAYER_ENABLE_FORCE_INPUT_PACK8)
            dst_elempack = 8;

        // pick another dst_elempack for testing any_packing feature
        int any_elempack = dst_elempack;
        if (op->support_any_packing)
        {
            if (elembits == 32)
            {
#if NCNN_AVX512
                if (elemcount % 16 == 0 && ncnn::cpu_support_x86_avx512())
                    any_elempack = 8;
                else if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                    any_elempack = 4;
                else if (elemcount % 4 == 0)
                    any_elempack = 1;
#elif NCNN_AVX
                if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                    any_elempack = 4;
                else if (elemcount % 4 == 0)
                    any_elempack = 1;
#elif NCNN_RVV || NCNN_XTHEADVECTOR
                const int packn = ncnn::cpu_riscv_vlenb() / 4;
                if (elemcount % packn == 0)
                    any_elempack = 1;
#else
                if (elemcount % 4 == 0)
                    any_elempack = 1;
#endif
            }
            if (elembits == 16)
            {
#if NCNN_ARM82
                if (elemcount % 8 == 0 && ncnn::cpu_support_arm_asimdhp() && opt.use_fp16_arithmetic && op->support_fp16_storage)
                    any_elempack = 4;
                else if (elemcount % 4 == 0)
                    any_elempack = 1;
#elif NCNN_RVV || NCNN_XTHEADVECTOR
                const int packn = ncnn::cpu_riscv_vlenb() / 2;
                if (elemcount % packn == 0)
                    any_elempack = 1;
#else
                if (elemcount % 4 == 0)
                    any_elempack = 1;
#endif
            }
            if (elembits == 8)
            {
#if NCNN_RVV || NCNN_XTHEADVECTOR
                const int packn = ncnn::cpu_riscv_vlenb() / 1;
                if (elemcount % packn == 0)
                    any_elempack = 1;
#else
                if (elemcount % 8 == 0)
                    any_elempack = 1;
#endif
            }

            if (flag & TEST_LAYER_ENABLE_FORCE_INPUT_PACK8)
                any_elempack = 8;
        }

        ncnn::Mat a4_packed;
        ncnn::convert_packing(a4, a4_packed, dst_elempack, opt);
        a4 = a4_packed;

        if (any_elempack != dst_elempack)
        {
            ncnn::Mat ax_packed;
            ncnn::convert_packing(ax, ax_packed, any_elempack, opt);
            ax = ax_packed;
        }
        else
        {
            ax = a4;
        }
    }

    return 0;
}

static int convert_to_vanilla_layout(const ncnn::Mat& c4, ncnn::Mat& c, const ncnn::Option& opt, const ncnn::Layer* op, int flag)
{
    ncnn::Mat c4_unpacked;
    if (c4.elempack != 1)
    {
        ncnn::convert_packing(c4, c4_unpacked, 1, opt);
    }
    else
    {
        c4_unpacked = c4;
    }

    // clang-format off
    // *INDENT-OFF*
#if NCNN_ARM82
    if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && op->support_fp16_storage && c4_unpacked.elembits() == 16)
    {
        ncnn::cast_float16_to_float32(c4_unpacked, c, opt);
    }
    else
#endif // NCNN_ARM82
#if NCNN_VFPV4
    if (opt.use_fp16_storage && !opt.use_bf16_storage && ncnn::cpu_support_arm_vfpv4() && op->support_fp16_storage && c4_unpacked.elembits() == 16)
    {
        ncnn::cast_float16_to_float32(c4_unpacked, c, opt);
    }
    else
#endif // NCNN_VFPV4
#if NCNN_ZFH
    if (opt.use_fp16_storage && (ncnn::cpu_support_riscv_zvfh() || (!ncnn::cpu_support_riscv_v() && ncnn::cpu_support_riscv_zfh())) && op->support_fp16_storage && c4_unpacked.elembits() == 16)
    {
        ncnn::cast_float16_to_float32(c4_unpacked, c, opt);
    }
    else
#endif // NCNN_ZFH
#if NCNN_BF16
    if (opt.use_bf16_storage && op->support_bf16_storage && c4_unpacked.elembits() == 16)
    {
        ncnn::cast_bfloat16_to_float32(c4_unpacked, c, opt);
    }
    else
#endif // NCNN_BF16
    if (opt.use_fp16_storage && op->support_fp16_storage && c4_unpacked.elembits() == 16)
    {
        ncnn::cast_float16_to_float32(c4_unpacked, c, opt);
    }
    else
    {
        c = c4_unpacked;
    }
    // *INDENT-ON*
    // clang-format on

    return 0;
}

int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& b, int flag)
{
    ncnn::Layer* op = ncnn::create_layer_naive(typeindex);

    op->load_param(pd);

    if (op->one_blob_only && a.size() != 1)
    {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = false;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    b.resize(top_blob_count);

    if (op->support_inplace)
    {
        for (size_t i = 0; i < a.size(); i++)
        {
            b[i] = a[i].clone();
        }

        op->forward_inplace(b, opt);
    }
    else
    {
        op->forward(a, b, opt);
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& c, const std::vector<ncnn::Mat>& top_shapes, int flag)
{
    ncnn::Layer* op = ncnn::create_layer_cpu(typeindex);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        delete op;
        return 233;
    }

    if (!top_shapes.empty())
    {
        op->bottom_shapes = a;
        op->top_shapes = top_shapes;
    }

    op->load_param(pd);

    if (op->one_blob_only && a.size() != 1)
    {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt = _opt;
    opt.use_vulkan_compute = false;

    if (flag & TEST_LAYER_ENABLE_THREADING)
        opt.num_threads = ncnn::get_physical_big_cpu_count();
    else
        opt.num_threads = 1;

    op->create_pipeline(opt);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }

    std::vector<ncnn::Mat> a4(a.size());
    std::vector<ncnn::Mat> ax(a.size());

    bool to_test_any_packing = false;
    for (size_t i = 0; i < a4.size(); i++)
    {
        convert_to_optimal_layout(a[i], a4[i], ax[i], opt, op, flag);

        if (ax[i].elempack != a4[i].elempack)
            to_test_any_packing = true;
    }

    if (!opt.use_packing_layout)
        to_test_any_packing = false;

    c.resize(top_blob_count);
    std::vector<ncnn::Mat> cx;
    cx.resize(top_blob_count);

    if (op->support_inplace)
    {
        for (size_t i = 0; i < a4.size(); i++)
        {
            c[i] = a4[i].clone();
        }

        op->forward_inplace(c, opt);
    }
    else
    {
        op->forward(a4, c, opt);
    }

    for (size_t i = 0; i < c.size(); i++)
    {
        convert_to_vanilla_layout(c[i], c[i], opt, op, flag);
    }

    if (to_test_any_packing)
    {
        if (op->support_inplace)
        {
            for (size_t i = 0; i < ax.size(); i++)
            {
                cx[i] = ax[i].clone();
            }

            op->forward_inplace(cx, opt);
        }
        else
        {
            op->forward(ax, cx, opt);
        }

        for (size_t i = 0; i < cx.size(); i++)
        {
            convert_to_vanilla_layout(cx[i], cx[i], opt, op, flag);
        }
    }

    op->destroy_pipeline(opt);

    delete op;

    if (to_test_any_packing)
    {
        float epsilon = 0.001f;
        if (opt.use_fp16_packed || opt.use_fp16_storage || opt.use_bf16_storage)
        {
            epsilon *= 100; // 0.1
        }

        for (size_t i = 0; i < cx.size(); i++)
        {
            if (CompareMat(c[i], cx[i], epsilon) != 0)
            {
                return -1;
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& d, const std::vector<ncnn::Mat>& top_shapes, int flag)
{
    if (!_opt.use_packing_layout)
    {
        // pack1 test is useless for gpu
        return 233;
    }

    ncnn::Layer* op = ncnn::create_layer_vulkan(typeindex);
    if (!op)
    {
        return 233;
    }

    op->load_param(pd);

    if (!op->support_vulkan)
    {
        delete op;
        return 233;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    op->vkdev = vkdev;

    if (!top_shapes.empty())
    {
        op->bottom_shapes = a;
        op->top_shapes = top_shapes;
    }

    if (op->one_blob_only && a.size() != 1)
    {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::VkWeightAllocator g_weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator g_weight_staging_vkallocator(vkdev);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = _opt;
    opt.use_vulkan_compute = true;

    if (flag & TEST_LAYER_ENABLE_THREADING)
        opt.num_threads = ncnn::get_physical_big_cpu_count();
    else
        opt.num_threads = 1;

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
    if (!vkdev->info.support_fp16_uniform()) opt.use_fp16_uniform = false;
    if (!vkdev->info.support_fp16_arithmetic()) opt.use_fp16_arithmetic = false;
    if (!vkdev->info.support_int8_packed()) opt.use_int8_packed = false;
    if (!vkdev->info.support_int8_storage()) opt.use_int8_storage = false;
    if (!vkdev->info.support_int8_uniform()) opt.use_int8_uniform = false;
    if (!vkdev->info.support_int8_arithmetic()) opt.use_int8_arithmetic = false;
    if (!vkdev->info.support_cooperative_matrix()) opt.use_cooperative_matrix = false;
    if (!vkdev->info.support_subgroup_ops()) opt.use_subgroup_ops = false;

    // FIXME fp16a may produce large error
    opt.use_fp16_arithmetic = false;

    op->create_pipeline(opt);

    if (!op->support_vulkan)
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }

    {
        ncnn::VkTransfer cmd(vkdev);

        ncnn::Option opt_upload = opt;
        opt_upload.blob_vkallocator = &g_weight_vkallocator;
        opt_upload.workspace_vkallocator = &g_weight_vkallocator;
        opt_upload.staging_vkallocator = &g_weight_staging_vkallocator;

        op->upload_model(cmd, opt_upload);

        cmd.submit_and_wait();
    }

    d.resize(top_blob_count);

    std::vector<ncnn::Mat> dx;
    dx.resize(top_blob_count);
    bool to_test_any_packing = false;
    {
        // forward
        ncnn::VkCompute cmd(vkdev);

        {
            std::vector<ncnn::VkMat> a_gpu(a.size());
            std::vector<ncnn::VkMat> ax_gpu(a.size());
            for (size_t i = 0; i < a_gpu.size(); i++)
            {
                int elemcount = 0;
                {
                    int dims = a[i].dims;
                    if (dims == 1) elemcount = a[i].elempack * a[i].w;
                    if (dims == 2) elemcount = a[i].elempack * a[i].h;
                    if (dims == 3 || dims == 4) elemcount = a[i].elempack * a[i].c;
                }

                // upload
                if (flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING)
                {
                    // resolve dst_elempack
                    const int dst_elempack = elemcount % 4 == 0 ? 4 : 1;

                    ncnn::Mat a4;
                    ncnn::convert_packing(a[i], a4, dst_elempack, opt);

                    ncnn::Option opt_upload = opt;
                    opt_upload.use_fp16_packed = false;
                    opt_upload.use_fp16_storage = false;
                    opt_upload.use_int8_packed = false;
                    opt_upload.use_int8_storage = false;

                    cmd.record_clone(a4, a_gpu[i], opt_upload);
                }
                else
                {
                    cmd.record_upload(a[i], a_gpu[i], opt);
                }

                // convert layout
                {
                    int dst_elempack = 1;
                    int any_elempack = 1;
                    if (op->support_vulkan_packing)
                    {
                        dst_elempack = elemcount % 4 == 0 ? 4 : 1;
                        any_elempack = dst_elempack;
                        if (op->support_vulkan_any_packing)
                        {
                            any_elempack = 1;
                        }
                    }

                    if (a_gpu[i].elempack != dst_elempack)
                    {
                        ncnn::VkMat a_gpu_packed;
                        vkdev->convert_packing(a_gpu[i], a_gpu_packed, dst_elempack, cmd, opt);
                        a_gpu[i] = a_gpu_packed;
                    }

                    ax_gpu[i] = a_gpu[i];
                    if (any_elempack != dst_elempack)
                    {
                        vkdev->convert_packing(a_gpu[i], ax_gpu[i], any_elempack, cmd, opt);
                    }
                }

                if (ax_gpu[i].elempack != a_gpu[i].elempack)
                    to_test_any_packing = true;
            }

            std::vector<ncnn::VkMat> d_gpu(top_blob_count);
            if (op->support_inplace)
            {
                op->forward_inplace(a_gpu, cmd, opt);

                d_gpu = a_gpu;
            }
            else
            {
                op->forward(a_gpu, d_gpu, cmd, opt);
            }

            // download
            for (size_t i = 0; i < d_gpu.size(); i++)
            {
                cmd.record_download(d_gpu[i], d[i], opt);
            }

            if (to_test_any_packing)
            {
                std::vector<ncnn::VkMat> dx_gpu(top_blob_count);
                if (op->support_inplace)
                {
                    op->forward_inplace(ax_gpu, cmd, opt);

                    dx_gpu = ax_gpu;
                }
                else
                {
                    op->forward(ax_gpu, dx_gpu, cmd, opt);
                }

                // download
                for (size_t i = 0; i < dx_gpu.size(); i++)
                {
                    cmd.record_download(dx_gpu[i], dx[i], opt);
                }
            }
        }

        cmd.submit_and_wait();
    }

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    g_weight_vkallocator.clear();
    g_weight_staging_vkallocator.clear();

    if (to_test_any_packing)
    {
        float epsilon = 0.001f;
        if (opt.use_fp16_packed || opt.use_fp16_storage || opt.use_bf16_storage)
        {
            epsilon *= 100; // 0.1
        }

        for (size_t i = 0; i < dx.size(); i++)
        {
            if (CompareMat(d[i], dx[i], epsilon) != 0)
            {
                return -1;
            }
        }
    }

    return 0;
}
#endif // NCNN_VULKAN

int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, const std::vector<ncnn::Mat>& top_shapes, float epsilon, int flag)
{
    // naive
    std::vector<ncnn::Mat> b;
    {
        int ret = test_layer_naive(typeindex, pd, weights, a, top_blob_count, b, flag);
        if (ret != 233 && ret != 0)
        {
            fprintf(stderr, "test_layer_naive failed\n");
            return -1;
        }
    }

    // cpu
    {
        std::vector<ncnn::Mat> c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, top_blob_count, c, std::vector<ncnn::Mat>(), flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, c, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_cpu failed\n");
            return -1;
        }
    }

    // cpu shape hint
    {
        std::vector<ncnn::Mat> c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, top_blob_count, c, b, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, c, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_cpu failed with shape hint\n");
            return -1;
        }
    }

#if NCNN_VULKAN
    // gpu
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING))
    {
        std::vector<ncnn::Mat> d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, top_blob_count, d, std::vector<ncnn::Mat>(), flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_gpu failed\n");
            return -1;
        }
    }

    // gpu shape hint
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING))
    {
        std::vector<ncnn::Mat> d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, top_blob_count, d, b, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_gpu failed with shape hint\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, ncnn::Mat& b, int flag)
{
    ncnn::Layer* op = ncnn::create_layer_naive(typeindex);

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = false;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    if (op->support_inplace)
    {
        b = a.clone();
        op->forward_inplace(b, opt);
    }
    else
    {
        op->forward(a, b, opt);
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& c, const ncnn::Mat& top_shape, int flag)
{
    ncnn::Layer* op = ncnn::create_layer_cpu(typeindex);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        delete op;
        return 233;
    }

    if (top_shape.dims)
    {
        op->bottom_shapes.resize(1);
        op->top_shapes.resize(1);
        op->bottom_shapes[0] = a;
        op->top_shapes[0] = top_shape;
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt = _opt;
    opt.use_vulkan_compute = false;

    if (flag & TEST_LAYER_ENABLE_THREADING)
        opt.num_threads = ncnn::get_physical_big_cpu_count();
    else
        opt.num_threads = 1;

    op->create_pipeline(opt);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }

    ncnn::Mat a4;
    ncnn::Mat ax;
    convert_to_optimal_layout(a, a4, ax, opt, op, flag);

    bool to_test_any_packing = ax.elempack != a4.elempack;

    if (!opt.use_packing_layout)
        to_test_any_packing = false;

    ncnn::Mat cx;

    if (op->support_inplace)
    {
        c = a4.clone();
        op->forward_inplace(c, opt);
    }
    else
    {
        op->forward(a4, c, opt);
    }

    convert_to_vanilla_layout(c, c, opt, op, flag);

    if (to_test_any_packing)
    {
        if (op->support_inplace)
        {
            cx = ax.clone();
            op->forward_inplace(cx, opt);
        }
        else
        {
            op->forward(ax, cx, opt);
        }

        convert_to_vanilla_layout(cx, cx, opt, op, flag);
    }

    op->destroy_pipeline(opt);

    delete op;

    if (to_test_any_packing)
    {
        float epsilon = 0.001f;
        if (opt.use_fp16_packed || opt.use_fp16_storage || opt.use_bf16_storage)
        {
            epsilon *= 100; // 0.1
        }

        if (CompareMat(c, cx, epsilon) != 0)
        {
            return -1;
        }
    }

    return 0;
}

#if NCNN_VULKAN
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& d, const ncnn::Mat& top_shape, int flag)
{
    if (!_opt.use_packing_layout)
    {
        // pack1 test is useless for gpu
        return 233;
    }

    ncnn::Layer* op = ncnn::create_layer_vulkan(typeindex);
    if (!op)
    {
        return 233;
    }

    op->load_param(pd);

    if (!op->support_vulkan)
    {
        delete op;
        return 233;
    }

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    op->vkdev = vkdev;

    if (top_shape.dims)
    {
        op->bottom_shapes.resize(1);
        op->top_shapes.resize(1);
        op->bottom_shapes[0] = a;
        op->top_shapes[0] = top_shape;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::VkWeightAllocator g_weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator g_weight_staging_vkallocator(vkdev);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = _opt;
    opt.use_vulkan_compute = true;

    if (flag & TEST_LAYER_ENABLE_THREADING)
        opt.num_threads = ncnn::get_physical_big_cpu_count();
    else
        opt.num_threads = 1;

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
    if (!vkdev->info.support_fp16_uniform()) opt.use_fp16_uniform = false;
    if (!vkdev->info.support_fp16_arithmetic()) opt.use_fp16_arithmetic = false;
    if (!vkdev->info.support_int8_packed()) opt.use_int8_packed = false;
    if (!vkdev->info.support_int8_storage()) opt.use_int8_storage = false;
    if (!vkdev->info.support_int8_uniform()) opt.use_int8_uniform = false;
    if (!vkdev->info.support_int8_arithmetic()) opt.use_int8_arithmetic = false;
    if (!vkdev->info.support_cooperative_matrix()) opt.use_cooperative_matrix = false;
    if (!vkdev->info.support_subgroup_ops()) opt.use_subgroup_ops = false;

    // FIXME fp16a may produce large error
    opt.use_fp16_arithmetic = false;

    op->create_pipeline(opt);

    if (!op->support_vulkan)
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }

    {
        ncnn::VkTransfer cmd(vkdev);

        ncnn::Option opt_upload = opt;
        opt_upload.blob_vkallocator = &g_weight_vkallocator;
        opt_upload.workspace_vkallocator = &g_weight_vkallocator;
        opt_upload.staging_vkallocator = &g_weight_staging_vkallocator;

        op->upload_model(cmd, opt_upload);

        cmd.submit_and_wait();
    }

    ncnn::Mat dx;
    bool to_test_any_packing = false;
    {
        // forward
        ncnn::VkCompute cmd(vkdev);

        {
            ncnn::VkMat a_gpu;
            ncnn::VkMat ax_gpu;

            int elemcount = 0;
            {
                int dims = a.dims;
                if (dims == 1) elemcount = a.elempack * a.w;
                if (dims == 2) elemcount = a.elempack * a.h;
                if (dims == 3 || dims == 4) elemcount = a.elempack * a.c;
            }

            // upload
            if (flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING)
            {
                // resolve dst_elempack
                const int dst_elempack = elemcount % 4 == 0 ? 4 : 1;

                ncnn::Mat a4;
                ncnn::convert_packing(a, a4, dst_elempack, opt);

                ncnn::Option opt_upload = opt;
                opt_upload.use_fp16_packed = false;
                opt_upload.use_fp16_storage = false;
                opt_upload.use_int8_packed = false;
                opt_upload.use_int8_storage = false;

                cmd.record_clone(a4, a_gpu, opt_upload);
            }
            else
            {
                cmd.record_upload(a, a_gpu, opt);
            }

            // convert layout
            {
                int dst_elempack = 1;
                int any_elempack = 1;
                if (op->support_vulkan_packing)
                {
                    dst_elempack = elemcount % 4 == 0 ? 4 : 1;
                    any_elempack = dst_elempack;
                    if (op->support_vulkan_any_packing)
                    {
                        any_elempack = 1;
                    }
                }

                if (a_gpu.elempack != dst_elempack)
                {
                    ncnn::VkMat a_gpu_packed;
                    vkdev->convert_packing(a_gpu, a_gpu_packed, dst_elempack, cmd, opt);
                    a_gpu = a_gpu_packed;
                }

                ax_gpu = a_gpu;
                if (any_elempack != dst_elempack)
                {
                    vkdev->convert_packing(a_gpu, ax_gpu, any_elempack, cmd, opt);
                }
            }

            if (ax_gpu.elempack != a_gpu.elempack)
                to_test_any_packing = true;

            ncnn::VkMat d_gpu;
            if (op->support_inplace)
            {
                op->forward_inplace(a_gpu, cmd, opt);

                d_gpu = a_gpu;
            }
            else
            {
                op->forward(a_gpu, d_gpu, cmd, opt);
            }

            // download
            cmd.record_download(d_gpu, d, opt);

            if (to_test_any_packing)
            {
                ncnn::VkMat dx_gpu;
                if (op->support_inplace)
                {
                    op->forward_inplace(ax_gpu, cmd, opt);

                    dx_gpu = ax_gpu;
                }
                else
                {
                    op->forward(ax_gpu, dx_gpu, cmd, opt);
                }

                // download
                cmd.record_download(dx_gpu, dx, opt);
            }
        }

        cmd.submit_and_wait();
    }

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    g_weight_vkallocator.clear();
    g_weight_staging_vkallocator.clear();

    if (to_test_any_packing)
    {
        float epsilon = 0.001f;
        if (opt.use_fp16_packed || opt.use_fp16_storage || opt.use_bf16_storage)
        {
            epsilon *= 100; // 0.1
        }

        if (CompareMat(d, dx, epsilon) != 0)
        {
            return -1;
        }
    }

    return 0;
}
#endif // NCNN_VULKAN

int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, const ncnn::Mat& top_shape, float epsilon, int flag)
{
    // naive
    ncnn::Mat b;
    {
        int ret = test_layer_naive(typeindex, pd, weights, a, b, flag);
        if (ret != 233 && ret != 0)
        {
            fprintf(stderr, "test_layer_naive failed\n");
            return -1;
        }
    }

    // cpu
    {
        ncnn::Mat c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, c, ncnn::Mat(), flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, c, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_cpu failed\n");
            return -1;
        }
    }

    // cpu shape hint
    {
        ncnn::Mat c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, c, b, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, c, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_cpu failed with shape hint\n");
            return -1;
        }
    }

#if NCNN_VULKAN
    // gpu
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING))
    {
        ncnn::Mat d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, d, ncnn::Mat(), flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_gpu failed\n");
            return -1;
        }
    }

    // gpu shape hint
    if (!(flag & TEST_LAYER_DISABLE_GPU_TESTING))
    {
        ncnn::Mat d;
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, d, b, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_gpu failed with shape hint\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

int test_layer_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const std::vector<ncnn::Mat>& a, int top_blob_count, float epsilon, int flag)
{
    // fp16 representation
    std::vector<ncnn::Mat> a_fp16;
    if (opt.use_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        a_fp16.resize(a.size());
        for (size_t j = 0; j < a.size(); j++)
        {
            ncnn::Mat tmp;
            ncnn::cast_float32_to_bfloat16(a[j], tmp, opt);
            ncnn::cast_bfloat16_to_float32(tmp, a_fp16[j], opt);
        }
    }
    else if ((opt.use_fp16_packed || opt.use_fp16_storage) && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        a_fp16.resize(a.size());
        for (size_t j = 0; j < a.size(); j++)
        {
            ncnn::Mat tmp;
            ncnn::cast_float32_to_float16(a[j], tmp, opt);
            ncnn::cast_float16_to_float32(tmp, a_fp16[j], opt);
        }
    }
    else
    {
        a_fp16 = a;
    }

    std::vector<ncnn::Mat> weights_fp16;
    float epsilon_fp16;
    if (opt.use_bf16_storage)
    {
        weights_fp16.resize(weights.size());
        for (size_t j = 0; j < weights.size(); j++)
        {
            if (weights[j].elembits() != 32)
            {
                weights_fp16[j] = weights[j];
                continue;
            }

            ncnn::Mat tmp;
            ncnn::cast_float32_to_bfloat16(weights[j], tmp, opt);
            ncnn::cast_bfloat16_to_float32(tmp, weights_fp16[j], opt);
        }
        epsilon_fp16 = epsilon * 100; // 0.1
    }
    else if (opt.use_fp16_packed || opt.use_fp16_storage)
    {
        weights_fp16.resize(weights.size());
        for (size_t j = 0; j < weights.size(); j++)
        {
            if (weights[j].elembits() != 32)
            {
                weights_fp16[j] = weights[j];
                continue;
            }

            ncnn::Mat tmp;
            ncnn::cast_float32_to_float16(weights[j], tmp, opt);
            ncnn::cast_float16_to_float32(tmp, weights_fp16[j], opt);
        }
        epsilon_fp16 = epsilon * 100; // 0.1
    }
    else
    {
        weights_fp16 = weights;
        epsilon_fp16 = epsilon;
    }

    if (opt.use_fp16_arithmetic)
    {
        epsilon_fp16 = epsilon * 1000; // 1.0
    }

    std::vector<ncnn::Mat> top_shapes;
    int ret = test_layer(ncnn::layer_to_index(layer_type), pd, weights_fp16, opt, a_fp16, top_blob_count, top_shapes, epsilon_fp16, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_fp16_arithmetic=%d use_bf16_storage=%d use_sgemm_convolution=%d use_winograd_convolution=%d\n", layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_fp16_arithmetic, opt.use_bf16_storage, opt.use_sgemm_convolution, opt.use_winograd_convolution);
        return ret;
    }

    return 0;
}

int test_layer_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const ncnn::Mat& a, float epsilon, int flag)
{
    // fp16 representation
    ncnn::Mat a_fp16;
    if (opt.use_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::Mat tmp;
        ncnn::cast_float32_to_bfloat16(a, tmp, opt);
        ncnn::cast_bfloat16_to_float32(tmp, a_fp16, opt);
    }
    else if ((opt.use_fp16_packed || opt.use_fp16_storage) && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::Mat tmp;
        ncnn::cast_float32_to_float16(a, tmp, opt);
        ncnn::cast_float16_to_float32(tmp, a_fp16, opt);
    }
    else
    {
        a_fp16 = a;
    }

    std::vector<ncnn::Mat> weights_fp16;
    float epsilon_fp16;
    if (opt.use_bf16_storage)
    {
        weights_fp16.resize(weights.size());
        for (size_t j = 0; j < weights.size(); j++)
        {
            if (weights[j].elembits() != 32)
            {
                weights_fp16[j] = weights[j];
                continue;
            }

            ncnn::Mat tmp;
            ncnn::cast_float32_to_bfloat16(weights[j], tmp, opt);
            ncnn::cast_bfloat16_to_float32(tmp, weights_fp16[j], opt);
        }
        epsilon_fp16 = epsilon * 100; // 0.1
    }
    else if (opt.use_fp16_packed || opt.use_fp16_storage)
    {
        weights_fp16.resize(weights.size());
        for (size_t j = 0; j < weights.size(); j++)
        {
            if (weights[j].elembits() != 32)
            {
                weights_fp16[j] = weights[j];
                continue;
            }

            ncnn::Mat tmp;
            ncnn::cast_float32_to_float16(weights[j], tmp, opt);
            ncnn::cast_float16_to_float32(tmp, weights_fp16[j], opt);
        }
        epsilon_fp16 = epsilon * 100; // 0.1
    }
    else
    {
        weights_fp16 = weights;
        epsilon_fp16 = epsilon;
    }

    if (opt.use_fp16_arithmetic)
    {
        epsilon_fp16 = epsilon * 1000; // 1.0
    }

    ncnn::Mat top_shape;
    int ret = test_layer(ncnn::layer_to_index(layer_type), pd, weights_fp16, opt, a_fp16, top_shape, epsilon_fp16, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_fp16_arithmetic=%d use_bf16_storage=%d use_sgemm_convolution=%d use_winograd_convolution=%d\n", layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_fp16_arithmetic, opt.use_bf16_storage, opt.use_sgemm_convolution, opt.use_winograd_convolution);
        return ret;
    }

    return 0;
}

int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count, float epsilon, int flag)
{
    // pack fp16p fp16s fp16a bf16s
    const int options[][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 0, 0, 1},
        {1, 0, 1, 0, 0},
        {1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1},
    };

    const int opt_count = sizeof(options) / sizeof(options[0]);

    for (int i = 0; i < opt_count; i++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = options[i][0];
        opt.use_fp16_packed = options[i][1];
        opt.use_fp16_storage = options[i][2];
        opt.use_fp16_arithmetic = options[i][3];
        opt.use_bf16_storage = options[i][4];

        int ret = test_layer_opt(layer_type, pd, weights, opt, a, top_blob_count, epsilon, flag);
        if (ret != 0)
            return ret;
    }

    return 0;
}

int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, float epsilon, int flag)
{
    // pack fp16p fp16s fp16a bf16s
    const int options[][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 0, 0, 1},
        {1, 0, 1, 0, 0},
        {1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1},
    };

    const int opt_count = sizeof(options) / sizeof(options[0]);

    for (int i = 0; i < opt_count; i++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = options[i][0];
        opt.use_fp16_packed = options[i][1];
        opt.use_fp16_storage = options[i][2];
        opt.use_fp16_arithmetic = options[i][3];
        opt.use_bf16_storage = options[i][4];

        int ret = test_layer_opt(layer_type, pd, weights, opt, a, epsilon, flag);
        if (ret != 0)
            return ret;
    }

    return 0;
}

class TestOOMAllocator : public ncnn::UnlockedPoolAllocator
{
public:
    TestOOMAllocator();
    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

    ncnn::Mutex lock;
    int counter;
    int failid;
};

TestOOMAllocator::TestOOMAllocator()
{
    counter = 0;
    failid = INT_MAX;
}

void* TestOOMAllocator::fastMalloc(size_t size)
{
    lock.lock();

    void* ptr;
    if (counter == failid)
    {
        ptr = 0;
    }
    else
    {
        ptr = ncnn::UnlockedPoolAllocator::fastMalloc(size);
    }
    counter++;

    lock.unlock();

    return ptr;
}

void TestOOMAllocator::fastFree(void* ptr)
{
    lock.lock();

    ncnn::UnlockedPoolAllocator::fastFree(ptr);

    lock.unlock();
}

int test_layer_oom_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, int flag)
{
    int typeindex = ncnn::layer_to_index(layer_type);
    if (typeindex == -1)
        return -1;

    ncnn::Layer* op = ncnn::create_layer_cpu(typeindex);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        delete op;
        return 233;
    }

    op->load_param(pd);

    if (op->one_blob_only && a.size() != 1)
    {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt = _opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }

    std::vector<ncnn::Mat> a4(a.size());
    std::vector<ncnn::Mat> ax(a.size());

    bool to_test_any_packing = false;
    for (size_t i = 0; i < a4.size(); i++)
    {
        convert_to_optimal_layout(a[i], a4[i], ax[i], opt, op, flag);

        if (ax[i].elempack != a4[i].elempack)
            to_test_any_packing = true;
    }

    TestOOMAllocator test_oom_allocator;
    opt.blob_allocator = &test_oom_allocator;
    opt.workspace_allocator = &test_oom_allocator;

    std::vector<ncnn::Mat> c;
    c.resize(top_blob_count);

    if (op->support_inplace)
    {
        for (size_t i = 0; i < a4.size(); i++)
        {
            c[i] = a4[i].clone();
        }

        op->forward_inplace(c, opt);
    }
    else
    {
        op->forward(a4, c, opt);
    }

    for (int i = 0; i < top_blob_count; i++)
    {
        c[i].release();
    }

    std::vector<ncnn::Mat> cx;
    cx.resize(top_blob_count);

    if (to_test_any_packing)
    {
        if (op->support_inplace)
        {
            for (size_t i = 0; i < ax.size(); i++)
            {
                cx[i] = ax[i].clone();
            }

            op->forward_inplace(cx, opt);
        }
        else
        {
            op->forward(ax, cx, opt);
        }

        for (int i = 0; i < top_blob_count; i++)
        {
            cx[i].release();
        }
    }

    const int alloc_count = test_oom_allocator.counter;
    for (int i = 0; i < alloc_count; i++)
    {
        test_oom_allocator.counter = 0;
        test_oom_allocator.failid = i;

        int ret = 0;
        if (op->support_inplace)
        {
            for (size_t i = 0; i < a4.size(); i++)
            {
                c[i] = a4[i].clone();
            }

            ret = op->forward_inplace(c, opt);
        }
        else
        {
            ret = op->forward(a4, c, opt);
        }

        for (int i = 0; i < top_blob_count; i++)
        {
            c[i].release();
        }

        if (ret == 0 && to_test_any_packing)
        {
            if (op->support_inplace)
            {
                for (size_t i = 0; i < ax.size(); i++)
                {
                    cx[i] = ax[i].clone();
                }

                ret = op->forward_inplace(cx, opt);
            }
            else
            {
                ret = op->forward(ax, cx, opt);
            }

            for (int i = 0; i < top_blob_count; i++)
            {
                cx[i].release();
            }
        }

        if (ret != -100)
        {
            fprintf(stderr, "oom not catched %d/%d\n", i, alloc_count);

            op->destroy_pipeline(opt);

            delete op;

            return -1;
        }
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

int test_layer_oom_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, int flag)
{
    int typeindex = ncnn::layer_to_index(layer_type);
    if (typeindex == -1)
        return -1;

    ncnn::Layer* op = ncnn::create_layer_cpu(typeindex);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        delete op;
        return 233;
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt = _opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    if (!op->support_packing && _opt.use_packing_layout)
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }
    if (!op->support_bf16_storage && !op->support_fp16_storage && (_opt.use_bf16_storage || _opt.use_fp16_arithmetic))
    {
        op->destroy_pipeline(opt);
        delete op;
        return 233;
    }

    ncnn::Mat a4;
    ncnn::Mat ax;
    convert_to_optimal_layout(a, a4, ax, opt, op, flag);

    bool to_test_any_packing = ax.elempack != a4.elempack;

    TestOOMAllocator test_oom_allocator;
    opt.blob_allocator = &test_oom_allocator;
    opt.workspace_allocator = &test_oom_allocator;

    ncnn::Mat c;
    ncnn::Mat cx;

    if (op->support_inplace)
    {
        c = a4.clone();
        op->forward_inplace(c, opt);
    }
    else
    {
        op->forward(a4, c, opt);
    }

    c.release();

    if (to_test_any_packing)
    {
        if (op->support_inplace)
        {
            cx = ax.clone();
            op->forward_inplace(cx, opt);
        }
        else
        {
            op->forward(ax, cx, opt);
        }

        cx.release();
    }

    const int alloc_count = test_oom_allocator.counter;
    for (int i = 0; i < alloc_count; i++)
    {
        test_oom_allocator.counter = 0;
        test_oom_allocator.failid = i;

        int ret = 0;
        if (op->support_inplace)
        {
            c = a4.clone();
            ret = op->forward_inplace(c, opt);
        }
        else
        {
            ret = op->forward(a4, c, opt);
        }

        c.release();

        if (ret == 0 && to_test_any_packing)
        {
            if (op->support_inplace)
            {
                cx = ax.clone();
                ret = op->forward_inplace(cx, opt);
            }
            else
            {
                ret = op->forward(ax, cx, opt);
            }

            cx.release();
        }

        if (ret != -100)
        {
            fprintf(stderr, "oom not catched %d/%d\n", i, alloc_count);

            op->destroy_pipeline(opt);

            delete op;

            return -1;
        }
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

int test_layer_oom(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count, int flag)
{
    // pack fp16p fp16s fp16a bf16s
    const int options[][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 0, 0, 1},
        {1, 0, 1, 0, 0},
        {1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1},
    };

    const int opt_count = sizeof(options) / sizeof(options[0]);

    for (int i = 0; i < opt_count; i++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = options[i][0];
        opt.use_fp16_packed = options[i][1];
        opt.use_fp16_storage = options[i][2];
        opt.use_fp16_arithmetic = options[i][3];
        opt.use_bf16_storage = options[i][4];

        int ret = test_layer_oom_opt(layer_type, pd, weights, opt, a, top_blob_count, flag);
        if (ret != 233 && ret != 0)
            return ret;
    }

    return 0;
}

int test_layer_oom(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, int flag)
{
    // pack fp16p fp16s fp16a bf16s
    const int options[][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 0, 0, 1},
        {1, 0, 1, 0, 0},
        {1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1},
    };

    const int opt_count = sizeof(options) / sizeof(options[0]);

    for (int i = 0; i < opt_count; i++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = options[i][0];
        opt.use_fp16_packed = options[i][1];
        opt.use_fp16_storage = options[i][2];
        opt.use_fp16_arithmetic = options[i][3];
        opt.use_bf16_storage = options[i][4];

        int ret = test_layer_oom_opt(layer_type, pd, weights, opt, a, flag);
        if (ret != 233 && ret != 0)
            return ret;
    }

    return 0;
}
