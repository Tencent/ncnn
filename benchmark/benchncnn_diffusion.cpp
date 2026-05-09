// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "gpu.h"
#include "net.h"

#include "benchncnn_diffusion_param_data.h"

#ifndef NCNN_SIMPLESTL
#include <algorithm>
#include <vector>
#endif

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* /*format*/, void* /*p*/) const
    {
        return 0;
    }

    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

namespace zimage_turbo {

struct ZImageTransformer
{
    ncnn::Net all_x_embedder;
    ncnn::Net noise_refiner;
    ncnn::Net unified_refiner;
    ncnn::Net all_final_layer;
};

static int load_net(ncnn::Net& net, const char* param_data, const ncnn::Option& opt)
{
    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

    int ret = net.load_param_mem(param_data);
    if (ret != 0)
        return ret;

    DataReaderFromEmpty dr;
    return net.load_model(dr);
}

static int load_zimage_transformer(ZImageTransformer& transformer, const ncnn::Option& opt)
{
    int ret = 0;
    ret = load_net(transformer.all_x_embedder, z_image_turbo_transformer_all_x_embedder_ncnn_param_data, opt);
    if (ret != 0)
        return ret;

    ret = load_net(transformer.noise_refiner, z_image_turbo_transformer_noise_refiner_ncnn_param_data, opt);
    if (ret != 0)
        return ret;

    ret = load_net(transformer.unified_refiner, z_image_turbo_transformer_unified_ncnn_param_data, opt);
    if (ret != 0)
        return ret;

    ret = load_net(transformer.all_final_layer, z_image_turbo_transformer_all_final_layer_ncnn_param_data, opt);
    if (ret != 0)
        return ret;

    return 0;
}

static void make_rope_cache(int half_dim, int seqlen, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache)
{
    cos_cache.create(half_dim, seqlen);
    sin_cache.create(half_dim, seqlen);

    cos_cache.fill(1.f);
    sin_cache.fill(0.f);
}

static void concat_along_h(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& out)
{
    const int w = a.w;
    out.create(w, a.h + b.h);

    memcpy(out, a, (size_t)w * a.h * sizeof(float));
    memcpy(out.row(a.h), b, (size_t)w * b.h * sizeof(float));
}

static int process_all_x_embedder(ncnn::Net& net, const ncnn::Mat& x, ncnn::Mat& x_embed)
{
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", x);
    return ex.extract("out0", x_embed);
}

static int process_noise_refiner(ncnn::Net& net, const ncnn::Mat& x_embed, const ncnn::Mat& x_cos, const ncnn::Mat& x_sin, const ncnn::Mat& t_embed, ncnn::Mat& x_embed_refine)
{
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", x_embed);
    ex.input("in1", x_cos);
    ex.input("in2", x_sin);
    ex.input("in3", t_embed);
    return ex.extract("out0", x_embed_refine);
}

static int process_unified_refiner(ncnn::Net& net, const ncnn::Mat& unified_embed, const ncnn::Mat& unified_cos, const ncnn::Mat& unified_sin, const ncnn::Mat& t_embed, ncnn::Mat& unified)
{
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", unified_embed);
    ex.input("in1", unified_cos);
    ex.input("in2", unified_sin);
    ex.input("in3", t_embed);
    return ex.extract("out0", unified);
}

static int process_all_final_layer(ncnn::Net& net, const ncnn::Mat& unified, const ncnn::Mat& t_embed, ncnn::Mat& unified_final)
{
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", unified);
    ex.input("in1", t_embed);
    return ex.extract("out0", unified_final);
}

static int run_zimage_512_step(ZImageTransformer& transformer, const ncnn::Mat& x0, const ncnn::Mat& cap_refine, const ncnn::Mat& x_cos, const ncnn::Mat& x_sin, const ncnn::Mat& unified_cos, const ncnn::Mat& unified_sin, const ncnn::Mat& t_embed)
{
    ncnn::Mat x = x0.clone();
    if (x.empty())
        return -100;

    ncnn::Mat x_embed;
    int ret = process_all_x_embedder(transformer.all_x_embedder, x, x_embed);
    if (ret != 0)
        return ret;

    ncnn::Mat x_embed_refine;
    ret = process_noise_refiner(transformer.noise_refiner, x_embed, x_cos, x_sin, t_embed, x_embed_refine);
    if (ret != 0)
        return ret;

    ncnn::Mat unified_embed;
    concat_along_h(x_embed_refine, cap_refine, unified_embed);
    if (unified_embed.empty())
        return -100;

    ncnn::Mat unified;
    ret = process_unified_refiner(transformer.unified_refiner, unified_embed, unified_cos, unified_sin, t_embed, unified);
    if (ret != 0)
        return ret;

    ncnn::Mat unified_final;
    ret = process_all_final_layer(transformer.all_final_layer, unified, t_embed, unified_final);
    if (ret != 0)
        return ret;

    const size_t total = x.total();
    for (size_t i = 0; i < total; i++)
    {
        x[i] = x[i] - unified_final[i];
    }

    return 0;
}

static void benchmark_512_step(const ncnn::Option& opt)
{
    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    ZImageTransformer transformer;
    int ret = load_zimage_transformer(transformer, opt);
    if (ret != 0)
    {
        fprintf(stderr, "zimage_turbo_6b_512_step load failed\n");
        return;
    }

    if (g_enable_cooling_down)
    {
        ncnn::sleep(10 * 1000);
    }

    const int image_tokens = 32 * 32;
    const int cap_tokens = 128;
    const int hidden_size = 3840;
    const int rope_half_dim = 64;

    ncnn::Mat x(64, image_tokens);
    x.fill(0.01f);

    ncnn::Mat cap_refine(hidden_size, cap_tokens);
    cap_refine.fill(0.01f);

    ncnn::Mat x_cos;
    ncnn::Mat x_sin;
    make_rope_cache(rope_half_dim, image_tokens, x_cos, x_sin);

    ncnn::Mat cap_cos;
    ncnn::Mat cap_sin;
    make_rope_cache(rope_half_dim, cap_tokens, cap_cos, cap_sin);

    ncnn::Mat unified_cos;
    ncnn::Mat unified_sin;
    concat_along_h(x_cos, cap_cos, unified_cos);
    concat_along_h(x_sin, cap_sin, unified_sin);

    ncnn::Mat t_embed(256);
    t_embed.fill(0.01f);

    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        run_zimage_512_step(transformer, x, cap_refine, x_cos, x_sin, unified_cos, unified_sin, t_embed);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();
        run_zimage_512_step(transformer, x, cap_refine, x_cos, x_sin, unified_cos, unified_sin, t_embed);
        double end = ncnn::get_current_time();

        double time = end - start;
        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    const double steps_per_second = 1000.0 / time_avg;
    fprintf(stderr, "%30s  min = %7.2f  max = %7.2f  avg = %7.2f  steps/s = %7.2f\n", "zimage_turbo_6b_512_step", time_min, time_max, time_avg, steps_per_second);
}

} // namespace zimage_turbo

static void show_usage()
{
    fprintf(stderr, "Usage: benchncnn_diffusion [loop count] [num threads] [powersave] [gpu device] [cooling down]\n");
}

int main(int argc, char** argv)
{
    int loop_count = 4;
    int num_threads = ncnn::get_physical_big_cpu_count();
    int powersave = 2;
    int gpu_device = -1;
    int cooling_down = 1;

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-' && argv[i][1] == 'h')
        {
            show_usage();
            return -1;
        }

        if (strcmp(argv[i], "--help") == 0)
        {
            show_usage();
            return -1;
        }
    }

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        gpu_device = atoi(argv[4]);
    }
    if (argc >= 6)
    {
        cooling_down = atoi(argv[5]);
    }

    const bool use_vulkan_compute = gpu_device != -1;

    g_enable_cooling_down = cooling_down != 0;
    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.f);

#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        g_warmup_loop_count = 10;

        g_vkdev = ncnn::get_gpu_device(gpu_device);

        g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    }
#endif // NCNN_VULKAN

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
#if NCNN_VULKAN
    opt.blob_vkallocator = g_blob_vkallocator;
    opt.workspace_vkallocator = g_blob_vkallocator;
    opt.staging_vkallocator = g_staging_vkallocator;
#endif // NCNN_VULKAN
    opt.use_winograd_convolution = false;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_vulkan_compute = use_vulkan_compute;
    opt.use_bf16_packed = true;
    opt.use_bf16_storage = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int)g_enable_cooling_down);

    zimage_turbo::benchmark_512_step(opt);

#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    return 0;
}
