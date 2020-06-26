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

#ifndef TESTUTIL_H
#define TESTUTIL_H

#include "layer.h"
#include "mat.h"
#include "prng.h"

#include <algorithm>
#include <math.h>
#include <stdio.h>

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND()      prng_rand(&g_prng_rand_state)

static float RandomFloat(float a = -2.f, float b = 2.f)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

static void Randomize(ncnn::Mat& m, float a = -2.f, float b = 2.f)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        m[i] = RandomFloat(a, b);
    }
}

static ncnn::Mat RandomMat(int w)
{
    ncnn::Mat m(w);
    Randomize(m);
    return m;
}

static ncnn::Mat RandomMat(int w, int h)
{
    ncnn::Mat m(w, h);
    Randomize(m);
    return m;
}

static ncnn::Mat RandomMat(int w, int h, int c)
{
    ncnn::Mat m(w, h, c);
    Randomize(m);
    return m;
}

static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

static int Compare(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001)
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
    CHECK_MEMBER(c)
    CHECK_MEMBER(elemsize)
    CHECK_MEMBER(elempack)

#undef CHECK_MEMBER

    for (int q = 0; q < a.c; q++)
    {
        const ncnn::Mat ma = a.channel(q);
        const ncnn::Mat mb = b.channel(q);
        for (int i = 0; i < a.h; i++)
        {
            const float* pa = ma.row(i);
            const float* pb = mb.row(i);
            for (int j = 0; j < a.w; j++)
            {
                if (!NearlyEqual(pa[j], pb[j], epsilon))
                {
                    fprintf(stderr, "value not match  at c:%d h:%d w:%d    expect %f but got %f\n", q, i, j, pa[j], pb[j]);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int CompareMat(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001)
{
    if (a.elempack != 1)
    {
        ncnn::Mat a1;
        ncnn::convert_packing(a, a1, 1);
        return CompareMat(a1, b, epsilon);
    }

    if (b.elempack != 1)
    {
        ncnn::Mat b1;
        ncnn::convert_packing(b, b1, 1);
        return CompareMat(a, b1, epsilon);
    }

    if (a.elemsize == 2u)
    {
        ncnn::Mat a32;
        cast_float16_to_float32(a, a32);
        return CompareMat(a32, b, epsilon);
    }
    if (a.elemsize == 1u)
    {
        ncnn::Mat a32;
        cast_int8_to_float32(a, a32);
        return CompareMat(a32, b, epsilon);
    }

    if (b.elemsize == 2u)
    {
        ncnn::Mat b32;
        cast_float16_to_float32(b, b32);
        return CompareMat(a, b32, epsilon);
    }
    if (b.elemsize == 1u)
    {
        ncnn::Mat b32;
        cast_int8_to_float32(b, b32);
        return CompareMat(a, b32, epsilon);
    }

    return Compare(a, b, epsilon);
}

static int CompareMat(const std::vector<ncnn::Mat>& a, const std::vector<ncnn::Mat>& b, float epsilon = 0.001)
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

template<typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, const std::vector<ncnn::Mat>& top_shapes = std::vector<ncnn::Mat>(), float epsilon = 0.001, void (*func)(T*) = 0)
{
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (func)
    {
        (*func)((T*)op);
    }

    ncnn::Option opt = _opt;

    if (!op->support_vulkan) opt.use_vulkan_compute = false;
    if (!op->support_packing) opt.use_packing_layout = false;
    if (!op->support_bf16_storage) opt.use_bf16_storage = false;
    if (!op->support_image_storage) opt.use_image_storage = false;

#if __APPLE__
    opt.use_image_storage = false;
#endif

    if (opt.use_int8_inference) opt.use_bf16_storage = false;
    if (opt.use_int8_inference) opt.use_packing_layout = false;

#if NCNN_VULKAN
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkWeightAllocator g_weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator g_weight_staging_vkallocator(vkdev);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    if (!vkdev->info.support_fp16_packed) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage) opt.use_fp16_storage = false;
    if (!vkdev->info.support_fp16_arithmetic) opt.use_fp16_arithmetic = false;

    op->vkdev = vkdev;
#endif // NCNN_VULKAN

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

    op->create_pipeline(opt);

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        ncnn::VkTransfer cmd(vkdev);

        ncnn::Option opt_upload = opt;
        opt_upload.blob_vkallocator = &g_weight_vkallocator;
        opt_upload.workspace_vkallocator = &g_weight_vkallocator;
        opt_upload.staging_vkallocator = &g_weight_staging_vkallocator;

        op->upload_model(cmd, opt_upload);

        cmd.submit_and_wait();
    }
#endif // NCNN_VULKAN

    std::vector<ncnn::Mat> b(top_blob_count);
    if (op->support_inplace)
    {
        for (size_t i = 0; i < a.size(); i++)
        {
            b[i] = a[i].clone();
        }

        ((T*)op)->T::forward_inplace(b, opt);
    }
    else
    {
        ((T*)op)->T::forward(a, b, opt);
    }

    std::vector<ncnn::Mat> c(top_blob_count);
    {
        std::vector<ncnn::Mat> a4(a.size());
        if (opt.use_packing_layout)
        {
            for (size_t i = 0; i < a.size(); i++)
            {
#if (defined(__x86_64__) || (defined _WIN32 && !(defined __MINGW32__)))
                ncnn::convert_packing(a[i], a4[i], 8, opt);
#else
                ncnn::convert_packing(a[i], a4[i], 4, opt);
#endif
            }
        }
        else
        {
            a4 = a;
        }

        if (opt.use_bf16_storage)
        {
            for (size_t i = 0; i < a4.size(); i++)
            {
                ncnn::Mat a_bf16;
                ncnn::cast_float32_to_bfloat16(a4[i], a_bf16, opt);
                a4[i] = a_bf16;
            }
        }

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

        if (opt.use_bf16_storage)
        {
            for (size_t i = 0; i < c.size(); i++)
            {
                ncnn::Mat c_fp32;
                ncnn::cast_bfloat16_to_float32(c[i], c_fp32, opt);
                c[i] = c_fp32;
            }
        }
    }

#if NCNN_VULKAN
    std::vector<ncnn::Mat> d(top_blob_count);
    if (opt.use_vulkan_compute)
    {
        // forward
        ncnn::VkCompute cmd(vkdev);

        if (opt.use_image_storage)
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
#endif // NCNN_VULKAN

    op->destroy_pipeline(opt);

    delete op;

#if NCNN_VULKAN
    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    g_weight_vkallocator.clear();
    g_weight_staging_vkallocator.clear();
#endif // NCNN_VULKAN

    if (CompareMat(b, c, epsilon) != 0)
    {
        fprintf(stderr, "test_layer failed cpu\n");
        return -1;
    }

#if NCNN_VULKAN
    if (opt.use_vulkan_compute && CompareMat(b, d, epsilon) != 0)
    {
        fprintf(stderr, "test_layer failed gpu\n");
        return -1;
    }
#endif // NCNN_VULKAN

    if (top_shapes.empty())
    {
        int ret = test_layer<T>(typeindex, pd, weights, opt, a, top_blob_count, b, epsilon, func);
        if (ret != 0)
        {
            fprintf(stderr, "test_layer failed gpu with shape hint\n");
        }
        return ret;
    }

    return 0;
}

template<typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, const ncnn::Mat& top_shape = ncnn::Mat(), float epsilon = 0.001, void (*func)(T*) = 0)
{
    ncnn::Layer* op = ncnn::create_layer(typeindex);
    ncnn::Option opt = _opt;

    if (func)
    {
        (*func)((T*)op);
    }

    if (!op->support_vulkan) opt.use_vulkan_compute = false;
    if (!op->support_packing) opt.use_packing_layout = false;
    if (!op->support_bf16_storage) opt.use_bf16_storage = false;
    if (!op->support_image_storage) opt.use_image_storage = false;

#if __APPLE__
    opt.use_image_storage = false;
#endif

    if (opt.use_int8_inference) opt.use_bf16_storage = false;
    if (opt.use_int8_inference) opt.use_packing_layout = false;

#if NCNN_VULKAN
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkWeightAllocator g_weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator g_weight_staging_vkallocator(vkdev);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    if (!vkdev->info.support_fp16_packed) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage) opt.use_fp16_storage = false;
    if (!vkdev->info.support_fp16_arithmetic) opt.use_fp16_arithmetic = false;

    op->vkdev = vkdev;
#endif // NCNN_VULKAN

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

    op->create_pipeline(opt);

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        ncnn::VkTransfer cmd(vkdev);

        ncnn::Option opt_upload = opt;
        opt_upload.blob_vkallocator = &g_weight_vkallocator;
        opt_upload.workspace_vkallocator = &g_weight_vkallocator;
        opt_upload.staging_vkallocator = &g_weight_staging_vkallocator;

        op->upload_model(cmd, opt_upload);

        cmd.submit_and_wait();
    }
#endif // NCNN_VULKAN

    ncnn::Mat b;
    if (op->support_inplace)
    {
        b = a.clone();
        ((T*)op)->T::forward_inplace(b, opt);
    }
    else
    {
        ((T*)op)->T::forward(a, b, opt);
    }

    ncnn::Mat c;
    {
        ncnn::Mat a4;
        if (opt.use_packing_layout)
        {
#if (defined(__x86_64__) || (defined _WIN32 && !(defined __MINGW32__)))
            ncnn::convert_packing(a, a4, 8, opt);
#else
            ncnn::convert_packing(a, a4, 4, opt);
#endif
        }
        else
        {
            a4 = a;
        }

        if (opt.use_bf16_storage)
        {
            ncnn::Mat a_bf16;
            ncnn::cast_float32_to_bfloat16(a4, a_bf16, opt);
            a4 = a_bf16;
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

        if (opt.use_bf16_storage)
        {
            ncnn::Mat c_fp32;
            ncnn::cast_bfloat16_to_float32(c, c_fp32, opt);
            c = c_fp32;
        }
    }

#if NCNN_VULKAN
    ncnn::Mat d;
    if (opt.use_vulkan_compute)
    {
        // forward
        ncnn::VkCompute cmd(vkdev);

        if (opt.use_image_storage)
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
#endif // NCNN_VULKAN

    op->destroy_pipeline(opt);

    delete op;

#if NCNN_VULKAN
    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);
    g_weight_vkallocator.clear();
    g_weight_staging_vkallocator.clear();
#endif // NCNN_VULKAN

    if (CompareMat(b, c, epsilon) != 0)
    {
        fprintf(stderr, "test_layer failed cpu\n");
        return -1;
    }

#if NCNN_VULKAN
    if (opt.use_vulkan_compute && CompareMat(b, d, epsilon) != 0)
    {
        fprintf(stderr, "test_layer failed gpu\n");
        return -1;
    }
#endif // NCNN_VULKAN

    if (top_shape.dims == 0)
    {
        int ret = test_layer<T>(typeindex, pd, weights, opt, a, b, epsilon, func);
        if (ret != 0)
        {
            fprintf(stderr, "test_layer failed gpu with shape hint\n");
        }
        return ret;
    }

    return 0;
}

template<typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, float epsilon = 0.001, void (*func)(T*) = 0)
{
    ncnn::Option opts[3];
    opts[0] = _opt;
    opts[0].use_packing_layout = false;
    opts[0].use_fp16_packed = false;
    opts[0].use_fp16_storage = false;
    opts[0].use_shader_pack8 = false;
    opts[0].use_image_storage = false;
    opts[1] = _opt;
    opts[1].use_packing_layout = true;
    opts[1].use_fp16_packed = true;
    opts[1].use_fp16_storage = false;
    opts[1].use_shader_pack8 = true;
    opts[1].use_image_storage = false;
    opts[2] = _opt;
    opts[2].use_packing_layout = true;
    opts[2].use_fp16_packed = true;
    opts[2].use_fp16_storage = true;
    opts[2].use_bf16_storage = true;
    opts[2].use_shader_pack8 = true;
    opts[2].use_image_storage = true;

    for (int i = 0; i < 3; i++)
    {
        const ncnn::Option& opt = opts[i];

        // fp16 representation
        std::vector<ncnn::Mat> a_fp16;
        std::vector<ncnn::Mat> weights_fp16;
        float epsilon_fp16;
        if (opt.use_bf16_storage)
        {
            a_fp16.resize(a.size());
            for (size_t j = 0; j < a.size(); j++)
            {
                ncnn::Mat tmp;
                ncnn::cast_float32_to_bfloat16(a[j], tmp, opt);
                ncnn::cast_bfloat16_to_float32(tmp, a_fp16[j], opt);
            }
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
            a_fp16.resize(a.size());
            for (size_t j = 0; j < a.size(); j++)
            {
                ncnn::Mat tmp;
                ncnn::cast_float32_to_float16(a[j], tmp, opt);
                ncnn::cast_float16_to_float32(tmp, a_fp16[j], opt);
            }
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
            a_fp16 = a;
            weights_fp16 = weights;
            epsilon_fp16 = epsilon;
        }

        std::vector<ncnn::Mat> top_shapes;
        int ret = test_layer<T>(ncnn::layer_to_index(layer_type), pd, weights_fp16, opt, a_fp16, top_blob_count, top_shapes, epsilon_fp16, func);
        if (ret != 0)
        {
            fprintf(stderr, "test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_shader_pack8=%d use_bf16_storage=%d use_image_storage=%d\n", layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_shader_pack8, opt.use_bf16_storage, opt.use_image_storage);
            return ret;
        }
    }

    return 0;
}

template<typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, float epsilon = 0.001, void (*func)(T*) = 0)
{
    ncnn::Option opts[3];
    opts[0] = _opt;
    opts[0].use_packing_layout = false;
    opts[0].use_fp16_packed = false;
    opts[0].use_fp16_storage = false;
    opts[0].use_shader_pack8 = false;
    opts[0].use_image_storage = false;
    opts[1] = _opt;
    opts[1].use_packing_layout = true;
    opts[1].use_fp16_packed = true;
    opts[1].use_fp16_storage = false;
    opts[1].use_shader_pack8 = true;
    opts[1].use_image_storage = false;
    opts[2] = _opt;
    opts[2].use_packing_layout = true;
    opts[2].use_fp16_packed = true;
    opts[2].use_fp16_storage = true;
    opts[2].use_bf16_storage = true;
    opts[2].use_shader_pack8 = true;
    opts[2].use_image_storage = true;

    for (int i = 0; i < 3; i++)
    {
        const ncnn::Option& opt = opts[i];

        // fp16 representation
        ncnn::Mat a_fp16;
        std::vector<ncnn::Mat> weights_fp16;
        float epsilon_fp16;
        if (opt.use_bf16_storage)
        {
            {
                ncnn::Mat tmp;
                ncnn::cast_float32_to_bfloat16(a, tmp, opt);
                ncnn::cast_bfloat16_to_float32(tmp, a_fp16, opt);
            }
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
            {
                ncnn::Mat tmp;
                ncnn::cast_float32_to_float16(a, tmp, opt);
                ncnn::cast_float16_to_float32(tmp, a_fp16, opt);
            }
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
            a_fp16 = a;
            weights_fp16 = weights;
            epsilon_fp16 = epsilon;
        }

        ncnn::Mat top_shape;
        int ret = test_layer<T>(ncnn::layer_to_index(layer_type), pd, weights_fp16, opt, a_fp16, top_shape, epsilon_fp16, func);
        if (ret != 0)
        {
            fprintf(stderr, "test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_shader_pack8=%d use_bf16_storage=%d use_image_storage=%d\n", layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_shader_pack8, opt.use_bf16_storage, opt.use_image_storage);
            return ret;
        }
    }

    return 0;
}

#endif // TESTUTIL_H
