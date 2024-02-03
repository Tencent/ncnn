// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "testutil.h"

#include "cpu.h"
#include "layer.h"
#include "mat.h"
#include "prng.h"

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

int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& b, void (*func)(ncnn::Layer*), int flag)
{
    ncnn::Layer* op = ncnn::create_layer_naive(typeindex);

    if (func)
    {
        (*func)((ncnn::Layer*)op);
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

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = false;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;
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

int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& c, const std::vector<ncnn::Mat>& top_shapes, void (*func)(ncnn::Layer*), int flag)
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

    if (func)
    {
        (*func)((ncnn::Layer*)op);
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

    for (size_t i = 0; i < a4.size(); i++)
    {
        // clang-format off
        // *INDENT-OFF*
#if NCNN_VFPV4
        if (opt.use_fp16_storage && ncnn::cpu_support_arm_vfpv4() && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
        {
            ncnn::cast_float32_to_float16(a[i], a4[i], opt);
        }
        else
#endif // NCNN_VFPV4
#if NCNN_RVV
        if (opt.use_fp16_storage && ncnn::cpu_support_riscv_v() && ncnn::cpu_support_riscv_zfh() && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
        {
            ncnn::cast_float32_to_float16(a[i], a4[i], opt);
        }
        else
#endif // NCNN_RVV
#if NCNN_BF16
        if (opt.use_bf16_storage && op->support_bf16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
        {
            ncnn::cast_float32_to_bfloat16(a[i], a4[i], opt);
        }
        else
#endif // NCNN_BF16
        if (opt.use_fp16_storage && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
        {
            ncnn::cast_float32_to_float16(a[i], a4[i], opt);
        }
        else
        {
            a4[i] = a[i];
        }
        // *INDENT-ON*
        // clang-format on

        if (opt.use_packing_layout && op->support_packing && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_PACKING))
        {
            // resolve dst_elempack
            int dims = a4[i].dims;
            int elemcount = 0;
            if (dims == 1) elemcount = a4[i].elempack * a4[i].w;
            if (dims == 2) elemcount = a4[i].elempack * a4[i].h;
            if (dims == 3 || dims == 4) elemcount = a4[i].elempack * a4[i].c;

            int elembits = a4[i].elembits();

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
#elif NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / (elembits / 8);
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
                if (elemcount % 8 == 0 && ncnn::cpu_support_arm_asimdhp() && opt.use_fp16_arithmetic)
                    dst_elempack = 8;
                else if (elemcount % 4 == 0)
                    dst_elempack = 4;
#elif NCNN_RVV
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
#if NCNN_RVV
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

            ncnn::Mat a4_packed;
            ncnn::convert_packing(a4[i], a4_packed, dst_elempack, opt);
            a4[i] = a4_packed;
        }
    }

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

    for (size_t i = 0; i < c.size(); i++)
    {
        // clang-format off
        // *INDENT-OFF*
#if NCNN_VFPV4
        if (opt.use_fp16_storage && ncnn::cpu_support_arm_vfpv4() && op->support_fp16_storage && c[i].elembits() == 16)
        {
            ncnn::Mat c_fp32;
            ncnn::cast_float16_to_float32(c[i], c_fp32, opt);
            c[i] = c_fp32;
        }
        else
#endif // NCNN_VFPV4
#if NCNN_RVV
        if (opt.use_fp16_storage && ncnn::cpu_support_riscv_v() && ncnn::cpu_support_riscv_zfh() && op->support_fp16_storage && c[i].elembits() == 16)
        {
            ncnn::Mat c_fp32;
            ncnn::cast_float16_to_float32(c[i], c_fp32, opt);
            c[i] = c_fp32;
        }
        else
#endif // NCNN_RVV
#if NCNN_BF16
        if (opt.use_bf16_storage && op->support_bf16_storage && c[i].elembits() == 16)
        {
            ncnn::Mat c_fp32;
            ncnn::cast_bfloat16_to_float32(c[i], c_fp32, opt);
            c[i] = c_fp32;
        }
        else
#endif // NCNN_BF16
        if (opt.use_fp16_storage && op->support_fp16_storage && c[i].elembits() == 16)
        {
            ncnn::Mat c_fp32;
            ncnn::cast_float16_to_float32(c[i], c_fp32, opt);
            c[i] = c_fp32;
        }
        // *INDENT-ON*
        // clang-format on
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if NCNN_VULKAN
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& d, const std::vector<ncnn::Mat>& top_shapes, void (*func)(ncnn::Layer*), int flag)
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

    if (func)
    {
        (*func)((ncnn::Layer*)op);
    }

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
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;

#if __APPLE__
    opt.use_image_storage = false;
#endif

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

    {
        // forward
        ncnn::VkCompute cmd(vkdev);

        if (op->support_image_storage && opt.use_image_storage)
        {
            // upload
            std::vector<ncnn::VkImageMat> a_gpu(a.size());
            for (size_t i = 0; i < a_gpu.size(); i++)
            {
                cmd.record_upload(a[i], a_gpu[i], opt);
            }

            std::vector<ncnn::VkImageMat> d_gpu(top_blob_count);
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
        }
        else
        {
            // upload
            std::vector<ncnn::VkMat> a_gpu(a.size());
            for (size_t i = 0; i < a_gpu.size(); i++)
            {
                cmd.record_upload(a[i], a_gpu[i], opt);
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
        }

        cmd.submit_and_wait();
    }

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    g_weight_vkallocator.clear();
    g_weight_staging_vkallocator.clear();

    return 0;
}
#endif // NCNN_VULKAN

int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, const std::vector<ncnn::Mat>& top_shapes, float epsilon, void (*func)(ncnn::Layer*), int flag)
{
    // naive
    std::vector<ncnn::Mat> b;
    {
        int ret = test_layer_naive(typeindex, pd, weights, a, top_blob_count, b, func, flag);
        if (ret != 233 && ret != 0)
        {
            fprintf(stderr, "test_layer_naive failed\n");
            return -1;
        }
    }

    // cpu
    {
        std::vector<ncnn::Mat> c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, top_blob_count, c, std::vector<ncnn::Mat>(), func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, c, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_cpu failed\n");
            return -1;
        }
    }

    // cpu shape hint
    {
        std::vector<ncnn::Mat> c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, top_blob_count, c, b, func, flag);
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
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, top_blob_count, d, std::vector<ncnn::Mat>(), func, flag);
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
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, top_blob_count, d, b, func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_gpu failed with shape hint\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, ncnn::Mat& b, void (*func)(ncnn::Layer*), int flag)
{
    ncnn::Layer* op = ncnn::create_layer_naive(typeindex);

    if (func)
    {
        (*func)((ncnn::Layer*)op);
    }

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
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;
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

int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& c, const ncnn::Mat& top_shape, void (*func)(ncnn::Layer*), int flag)
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

    if (func)
    {
        (*func)((ncnn::Layer*)op);
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

    // clang-format off
    // *INDENT-OFF*
#if NCNN_VFPV4
    if (opt.use_fp16_storage && ncnn::cpu_support_arm_vfpv4() && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::cast_float32_to_float16(a, a4, opt);
    }
    else
#endif // NCNN_VFPV4
#if NCNN_RVV
    if (opt.use_fp16_storage && ncnn::cpu_support_riscv_v() && ncnn::cpu_support_riscv_zfh() && op->support_fp16_storage && !(flag & TEST_LAYER_DISABLE_AUTO_INPUT_CASTING))
    {
        ncnn::cast_float32_to_float16(a, a4, opt);
    }
    else
#endif // NCNN_RVV
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
#elif NCNN_RVV
            const int packn = ncnn::cpu_riscv_vlenb() / (elembits / 8);
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
            if (elemcount % 8 == 0 && ncnn::cpu_support_arm_asimdhp() && opt.use_fp16_arithmetic)
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#elif NCNN_RVV
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
#if NCNN_RVV
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

        ncnn::Mat a4_packed;
        ncnn::convert_packing(a4, a4_packed, dst_elempack, opt);
        a4 = a4_packed;
    }

    if (op->support_inplace)
    {
        c = a4.clone();
        op->forward_inplace(c, opt);
    }
    else
    {
        op->forward(a4, c, opt);
    }

    // clang-format off
    // *INDENT-OFF*
#if NCNN_VFPV4
    if (opt.use_fp16_storage && ncnn::cpu_support_arm_vfpv4() && op->support_fp16_storage && c.elembits() == 16)
    {
        ncnn::Mat c_fp32;
        ncnn::cast_float16_to_float32(c, c_fp32, opt);
        c = c_fp32;
    }
    else
#endif // NCNN_VFPV4
#if NCNN_RVV
    if (opt.use_fp16_storage && ncnn::cpu_support_riscv_v() && ncnn::cpu_support_riscv_zfh() && op->support_fp16_storage && c.elembits() == 16)
    {
        ncnn::Mat c_fp32;
        ncnn::cast_float16_to_float32(c, c_fp32, opt);
        c = c_fp32;
    }
    else
#endif // NCNN_RVV
#if NCNN_BF16
    if (opt.use_bf16_storage && op->support_bf16_storage && c.elembits() == 16)
    {
        ncnn::Mat c_fp32;
        ncnn::cast_bfloat16_to_float32(c, c_fp32, opt);
        c = c_fp32;
    }
    else
#endif // NCNN_BF16
    if (opt.use_fp16_storage && op->support_fp16_storage && c.elembits() == 16)
    {
        ncnn::Mat c_fp32;
        ncnn::cast_float16_to_float32(c, c_fp32, opt);
        c = c_fp32;
    }
    // *INDENT-ON*
    // clang-format on

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if NCNN_VULKAN
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& d, const ncnn::Mat& top_shape, void (*func)(ncnn::Layer*), int flag)
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

    if (func)
    {
        (*func)((ncnn::Layer*)op);
    }

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
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;

#if __APPLE__
    opt.use_image_storage = false;
#endif

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

    {
        // forward
        ncnn::VkCompute cmd(vkdev);

        if (op->support_image_storage && opt.use_image_storage)
        {
            // upload
            ncnn::VkImageMat a_gpu;
            cmd.record_upload(a, a_gpu, opt);

            ncnn::VkImageMat d_gpu;
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
        }
        else
        {
            // upload
            ncnn::VkMat a_gpu;
            cmd.record_upload(a, a_gpu, opt);

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
        }

        cmd.submit_and_wait();
    }

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    g_weight_vkallocator.clear();
    g_weight_staging_vkallocator.clear();

    return 0;
}
#endif // NCNN_VULKAN

int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, const ncnn::Mat& top_shape, float epsilon, void (*func)(ncnn::Layer*), int flag)
{
    // naive
    ncnn::Mat b;
    {
        int ret = test_layer_naive(typeindex, pd, weights, a, b, func, flag);
        if (ret != 233 && ret != 0)
        {
            fprintf(stderr, "test_layer_naive failed\n");
            return -1;
        }
    }

    // cpu
    {
        ncnn::Mat c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, c, ncnn::Mat(), func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, c, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_cpu failed\n");
            return -1;
        }
    }

    // cpu shape hint
    {
        ncnn::Mat c;
        int ret = test_layer_cpu(typeindex, pd, weights, _opt, a, c, b, func, flag);
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
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, d, ncnn::Mat(), func, flag);
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
        int ret = test_layer_gpu(typeindex, pd, weights, _opt, a, d, b, func, flag);
        if (ret != 233 && (ret != 0 || CompareMat(b, d, epsilon) != 0))
        {
            fprintf(stderr, "test_layer_gpu failed with shape hint\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

int test_layer_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const std::vector<ncnn::Mat>& a, int top_blob_count, float epsilon, void (*func)(ncnn::Layer*), int flag)
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
    int ret = test_layer(ncnn::layer_to_index(layer_type), pd, weights_fp16, opt, a_fp16, top_blob_count, top_shapes, epsilon_fp16, func, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_fp16_arithmetic=%d use_shader_pack8=%d use_bf16_storage=%d use_image_storage=%d use_sgemm_convolution=%d use_winograd_convolution=%d\n", layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_fp16_arithmetic, opt.use_shader_pack8, opt.use_bf16_storage, opt.use_image_storage, opt.use_sgemm_convolution, opt.use_winograd_convolution);
        return ret;
    }

    return 0;
}

int test_layer_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const ncnn::Mat& a, float epsilon, void (*func)(ncnn::Layer*), int flag)
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
    int ret = test_layer(ncnn::layer_to_index(layer_type), pd, weights_fp16, opt, a_fp16, top_shape, epsilon_fp16, func, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_fp16_arithmetic=%d use_shader_pack8=%d use_bf16_storage=%d use_image_storage=%d use_sgemm_convolution=%d use_winograd_convolution=%d\n", layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_fp16_arithmetic, opt.use_shader_pack8, opt.use_bf16_storage, opt.use_image_storage, opt.use_sgemm_convolution, opt.use_winograd_convolution);
        return ret;
    }

    return 0;
}

int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count, float epsilon, void (*func)(ncnn::Layer*), int flag)
{
    // pack fp16p fp16s fp16a bf16s shader8 image
    const int options[][7] = {
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0},
        {1, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 1, 0, 0},
        {1, 0, 1, 0, 0, 1, 0},
        {1, 1, 1, 1, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1},
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
        opt.use_shader_pack8 = options[i][5];
        opt.use_image_storage = options[i][6];

        int ret = test_layer_opt(layer_type, pd, weights, opt, a, top_blob_count, epsilon, func, flag);
        if (ret != 0)
            return ret;
    }

    return 0;
}

int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, float epsilon, void (*func)(ncnn::Layer*), int flag)
{
    // pack fp16p fp16s fp16a bf16s shader8 image
    const int options[][7] = {
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0},
        {1, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 1, 0, 0},
        {1, 0, 1, 0, 0, 1, 0},
        {1, 1, 1, 1, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1},
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
        opt.use_shader_pack8 = options[i][5];
        opt.use_image_storage = options[i][6];

        int ret = test_layer_opt(layer_type, pd, weights, opt, a, epsilon, func, flag);
        if (ret != 0)
            return ret;
    }

    return 0;
}
