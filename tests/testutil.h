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

#include <math.h>
#include <stdio.h>

#include <algorithm>

#include "prng.h"

#include "mat.h"
#include "layer.h"

#if NCNN_VULKAN
#include "gpu.h"
#include "command.h"

class GlobalGpuInstance
{
public:
    GlobalGpuInstance() { ncnn::create_gpu_instance(); }
    ~GlobalGpuInstance() { ncnn::destroy_gpu_instance(); }
};
// initialize vulkan runtime before main()
GlobalGpuInstance g_global_gpu_instance;
#endif // NCNN_VULKAN

static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND() prng_rand(&g_prng_rand_state)

static float RandomFloat(float a = -2.f, float b = 2.f)
{
    float random = ((float) RAND()) / (float) uint64_t(-1);//RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

static void Randomize(ncnn::Mat& m, float a = -2.f, float b = 2.f)
{
    for (size_t i=0; i<m.total(); i++)
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
#define CHECK_MEMBER(m) \
    if (a.m != b.m) \
    { \
        fprintf(stderr, #m" not match    expect %d but got %d\n", (int)a.m, (int)b.m); \
        return -1; \
    }

    CHECK_MEMBER(dims)
    CHECK_MEMBER(w)
    CHECK_MEMBER(h)
    CHECK_MEMBER(c)
    CHECK_MEMBER(elemsize)
    CHECK_MEMBER(elempack)

#undef CHECK_MEMBER

    for (int q=0; q<a.c; q++)
    {
        const ncnn::Mat ma = a.channel(q);
        const ncnn::Mat mb = b.channel(q);
        for (int i=0; i<a.h; i++)
        {
            const float* pa = ma.row(i);
            const float* pb = mb.row(i);
            for (int j=0; j<a.w; j++)
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

    for (size_t i=0; i<a.size(); i++)
    {
        if (CompareMat(a[i], b[i], epsilon))
        {
            fprintf(stderr, "output blob %zu not match\n", i);
            return -1;
        }
    }

    return 0;
}

template <typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, const std::vector<ncnn::Mat>& top_shapes = std::vector<ncnn::Mat>(), float epsilon = 0.001, void (*func)(T*) = 0)
{
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (func)
    {
        (*func)((T*)op);
    }

    ncnn::Option opt = _opt;

    if (!op->support_packing) opt.use_packing_layout = false;

#if NCNN_VULKAN
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkWeightBufferAllocator g_weight_vkallocator(vkdev);
    ncnn::VkWeightStagingBufferAllocator g_weight_staging_vkallocator(vkdev);

    ncnn::VkBlobBufferAllocator g_blob_vkallocator(vkdev);
    ncnn::VkStagingBufferAllocator g_staging_vkallocator(vkdev);

    opt.blob_vkallocator = &g_blob_vkallocator;
    opt.workspace_vkallocator = &g_blob_vkallocator;
    opt.staging_vkallocator = &g_staging_vkallocator;

    if (!vkdev->info.support_fp16_storage) opt.use_fp16_storage = false;
    if (!vkdev->info.support_fp16_packed) opt.use_fp16_packed = false;

    op->vkdev = vkdev;
#endif // NCNN_VULKAN

    if (op->one_blob_only && a.size() != 1)
    {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    if (!top_shapes.empty())
    {
        op->bottom_shapes = a;
        op->top_shapes = top_shapes;
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        ncnn::VkTransfer cmd(vkdev);
        cmd.weight_vkallocator = &g_weight_vkallocator;
        cmd.staging_vkallocator = &g_weight_staging_vkallocator;

        op->upload_model(cmd, opt);

        cmd.submit_and_wait();
    }
#endif // NCNN_VULKAN

    std::vector<ncnn::Mat> b(top_blob_count);
    if (op->support_inplace)
    {
        for (size_t i=0; i<a.size(); i++)
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
            for (size_t i=0; i<a.size(); i++)
            {
                ncnn::convert_packing(a[i], a4[i], 4, opt);
            }
        }
        else
        {
            a4 = a;
        }

        if (op->support_inplace)
        {
            for (size_t i=0; i<a4.size(); i++)
            {
                c[i] = a4[i].clone();
            }

            op->forward_inplace(c, opt);
        }
        else
        {
            op->forward(a4, c, opt);
        }
    }

#if NCNN_VULKAN
    std::vector<ncnn::Mat> d(top_blob_count);
    if (opt.use_vulkan_compute)
    {
        // pack
        std::vector<ncnn::Mat> a4(a.size());
        for (size_t i=0; i<a.size(); i++)
        {
            if (opt.use_shader_pack8)
            {
                ncnn::convert_packing(a[i], a4[i], 8, opt);
                if (a4[i].elempack == 1)
                    ncnn::convert_packing(a[i], a4[i], 4, opt);
            }
            else
                ncnn::convert_packing(a[i], a4[i], 4, opt);
        }

        // fp16
        std::vector<ncnn::Mat> a4_fp16(a4.size());
        for (size_t i=0; i<a4.size(); i++)
        {
            if (opt.use_fp16_storage || ((a4[i].elempack == 4 || a4[i].elempack == 8) && opt.use_fp16_packed))
            {
                ncnn::cast_float32_to_float16(a4[i], a4_fp16[i], opt);
            }
            else
            {
                a4_fp16[i] = a4[i];
            }
        }

        // upload
        std::vector<ncnn::VkMat> a4_fp16_gpu(a4_fp16.size());
        for (size_t i=0; i<a4_fp16.size(); i++)
        {
            a4_fp16_gpu[i].create_like(a4_fp16[i], &g_blob_vkallocator, &g_staging_vkallocator);
            a4_fp16_gpu[i].prepare_staging_buffer();
            a4_fp16_gpu[i].upload(a4_fp16[i]);
        }

        // forward
        ncnn::VkCompute cmd(vkdev);

        for (size_t i=0; i<a4_fp16_gpu.size(); i++)
        {
            cmd.record_upload(a4_fp16_gpu[i]);
        }

        std::vector<ncnn::VkMat> d4_fp16_gpu(top_blob_count);
        if (op->support_inplace)
        {
            for (size_t i=0; i<a4_fp16_gpu.size(); i++)
            {
                d4_fp16_gpu[i].create_like(a4_fp16_gpu[i], a4_fp16_gpu[i].allocator, a4_fp16_gpu[i].staging_allocator);
                cmd.record_clone(a4_fp16_gpu[i], d4_fp16_gpu[i]);
            }

            op->forward_inplace(d4_fp16_gpu, cmd, opt);
        }
        else
        {
            op->forward(a4_fp16_gpu, d4_fp16_gpu, cmd, opt);
        }

        for (size_t i=0; i<d4_fp16_gpu.size(); i++)
        {
            d4_fp16_gpu[i].prepare_staging_buffer();
        }

        for (size_t i=0; i<d4_fp16_gpu.size(); i++)
        {
            cmd.record_download(d4_fp16_gpu[i]);
        }

        cmd.submit_and_wait();

        // download
        for (size_t i=0; i<d4_fp16_gpu.size(); i++)
        {
            d[i].create_like(d4_fp16_gpu[i]);
            d4_fp16_gpu[i].download(d[i]);
        }
    }
#endif // NCNN_VULKAN

    op->destroy_pipeline(opt);

    delete op;

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
        return test_layer<T>(typeindex, pd, weights, opt, a, top_blob_count, b, epsilon, func);

    return 0;
}

template <typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, const ncnn::Mat& top_shape = ncnn::Mat(), float epsilon = 0.001, void (*func)(T*) = 0)
{
    ncnn::Layer* op = ncnn::create_layer(typeindex);
    ncnn::Option opt = _opt;

    if (func)
    {
        (*func)((T*)op);
    }

    if (!op->support_packing) opt.use_packing_layout = false;

#if NCNN_VULKAN
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkWeightBufferAllocator g_weight_vkallocator(vkdev);
    ncnn::VkWeightStagingBufferAllocator g_weight_staging_vkallocator(vkdev);

    ncnn::VkBlobBufferAllocator g_blob_vkallocator(vkdev);
    ncnn::VkStagingBufferAllocator g_staging_vkallocator(vkdev);

    opt.blob_vkallocator = &g_blob_vkallocator;
    opt.workspace_vkallocator = &g_blob_vkallocator;
    opt.staging_vkallocator = &g_staging_vkallocator;

    if (!vkdev->info.support_fp16_storage) opt.use_fp16_storage = false;
    if (!vkdev->info.support_fp16_packed) opt.use_fp16_packed = false;

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
        cmd.weight_vkallocator = &g_weight_vkallocator;
        cmd.staging_vkallocator = &g_weight_staging_vkallocator;

        op->upload_model(cmd, opt);

        cmd.submit_and_wait();

        g_weight_staging_vkallocator.clear();
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
            ncnn::convert_packing(a, a4, 4, opt);
        }
        else
        {
            a4 = a;
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
    }

#if NCNN_VULKAN
    ncnn::Mat d;
    if (opt.use_vulkan_compute)
    {
        // pack
        ncnn::Mat a4;
        if (opt.use_shader_pack8)
        {
            ncnn::convert_packing(a, a4, 8, opt);
            if (a4.elempack != 8)
                ncnn::convert_packing(a, a4, 4, opt);
        }
        else
            ncnn::convert_packing(a, a4, 4, opt);

        // fp16
        ncnn::Mat a4_fp16;
        if (opt.use_fp16_storage || ((a4.elempack == 4 || a4.elempack == 8) && opt.use_fp16_packed))
        {
            ncnn::cast_float32_to_float16(a4, a4_fp16, opt);
        }
        else
        {
            a4_fp16 = a4;
        }

        // upload
        ncnn::VkMat a4_fp16_gpu;
        a4_fp16_gpu.create_like(a4_fp16, &g_blob_vkallocator, &g_staging_vkallocator);
        a4_fp16_gpu.prepare_staging_buffer();
        a4_fp16_gpu.upload(a4_fp16);

        // forward
        ncnn::VkCompute cmd(vkdev);

        cmd.record_upload(a4_fp16_gpu);

        ncnn::VkMat d4_fp16_gpu;
        if (op->support_inplace)
        {
            d4_fp16_gpu.create_like(a4_fp16_gpu, a4_fp16_gpu.allocator, a4_fp16_gpu.staging_allocator);
            cmd.record_clone(a4_fp16_gpu, d4_fp16_gpu);
            op->forward_inplace(d4_fp16_gpu, cmd, opt);
        }
        else
        {
            op->forward(a4_fp16_gpu, d4_fp16_gpu, cmd, opt);
        }

        d4_fp16_gpu.prepare_staging_buffer();

        cmd.record_download(d4_fp16_gpu);

        cmd.submit_and_wait();

        // download
        d.create_like(d4_fp16_gpu);
        d4_fp16_gpu.download(d);
    }
#endif // NCNN_VULKAN

    op->destroy_pipeline(opt);

    delete op;

#if NCNN_VULKAN
    g_blob_vkallocator.clear();
    g_staging_vkallocator.clear();
    g_weight_vkallocator.clear();
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
        return test_layer<T>(typeindex, pd, weights, opt, a, b, epsilon, func);

    return 0;
}

template <typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, float epsilon = 0.001, void (*func)(T*) = 0)
{
    ncnn::Option opts[2];
    opts[0] = _opt;
    opts[0].use_packing_layout = false;
    opts[1] = _opt;
    opts[1].use_packing_layout = true;

    for (int i = 0; i < 2; i++)
    {
        const ncnn::Option& opt = opts[i];

        std::vector<ncnn::Mat> top_shapes;
        int ret = test_layer<T>(ncnn::layer_to_index(layer_type), pd, weights, opt, a, top_blob_count, top_shapes, epsilon, func);
        if (ret != 0)
        {
            fprintf(stderr, "test_layer %s failed use_packing_layout=%d\n", layer_type, opt.use_packing_layout);
            return ret;
        }
    }

    return 0;
}

template <typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, float epsilon = 0.001, void (*func)(T*) = 0)
{
    ncnn::Option opts[2];
    opts[0] = _opt;
    opts[0].use_packing_layout = false;
    opts[1] = _opt;
    opts[1].use_packing_layout = true;

    for (int i = 0; i < 2; i++)
    {
        const ncnn::Option& opt = opts[i];

        ncnn::Mat top_shape;
        int ret = test_layer<T>(ncnn::layer_to_index(layer_type), pd, weights, opt, a, top_shape, epsilon, func);
        if (ret != 0)
        {
            fprintf(stderr, "test_layer %s failed use_packing_layout=%d\n", layer_type, opt.use_packing_layout);
            return ret;
        }
    }

    return 0;
}

#endif // TESTUTIL_H
