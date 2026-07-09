// Benchmark comparing GPU default (pack=1 baseline) vs GPU optimized (pack4+P2 larger tile)
// Usage: bench_mha_compare [loop_count] [num_threads] [gpu_device]

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "benchmark.h"
#include "cpu.h"
#include "gpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "modelbin.h"
#include "paramdict.h"

#include <vector>

#if NCNN_VULKAN
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

static int g_loop_count = 4;
static int g_warmup_loop_count = 2;

static float randf()
{
    return (float)rand() / (float)RAND_MAX * 2.f - 1.f;
}

void benchmark_mha(
    const char* name,
    int embed_dim,
    int num_heads,
    int seq_len,
    const ncnn::Option& opt)
{
    // Prepare random inputs
    ncnn::Mat q(embed_dim, seq_len);
    ncnn::Mat k(embed_dim, seq_len);
    ncnn::Mat v(embed_dim, seq_len);

    for (int i = 0; i < q.total(); i++) q[i] = randf();
    for (int i = 0; i < k.total(); i++) k[i] = randf();
    for (int i = 0; i < v.total(); i++) v[i] = randf();

    // Prepare layer parameters
    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * embed_dim);
    pd.set(3, embed_dim);
    pd.set(4, embed_dim);
    pd.set(5, 0);

    // Random weights
    std::vector<ncnn::Mat> weights(8);
    for (int i = 0; i < 8; i++)
    {
        int size = (i % 2 == 0) ? embed_dim * embed_dim : embed_dim;
        weights[i] = ncnn::Mat(size);
        for (int j = 0; j < weights[i].total(); j++)
            weights[i][j] = randf() * 0.1f;
    }

    // Create layer
    ncnn::Layer* layer = ncnn::create_layer(ncnn::LayerType::MultiHeadAttention);
    layer->load_param(pd);
    ncnn::ModelBinFromMatArray model_bin(weights.data());
    layer->load_model(model_bin);
    layer->create_pipeline(opt);

    // Warm-up
    std::vector<ncnn::Mat> bottom_blobs;
    bottom_blobs.push_back(q);
    bottom_blobs.push_back(k);
    bottom_blobs.push_back(v);
    std::vector<ncnn::Mat> top_blobs(1);
    for (int i = 0; i < g_warmup_loop_count; i++)
        layer->forward(bottom_blobs, top_blobs, opt);

    // Benchmark
    double total_ms = 0.0;
    double min_ms = DBL_MAX;
    double max_ms = 0.0;
    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();
        layer->forward(bottom_blobs, top_blobs, opt);
        double end = ncnn::get_current_time();
        double ms = end - start;
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    layer->destroy_pipeline(opt);
    delete layer;

    double avg_ms = total_ms / (double)g_loop_count;
    printf("  %-32s embed=%-4d heads=%-3d seq=%-4d  min=%8.2f ms  max=%8.2f ms  avg=%8.2f ms\n",
           name, embed_dim, num_heads, seq_len, min_ms, max_ms, avg_ms);
}

int main(int argc, char** argv)
{
    int loop_count = 10;
    int num_threads = 4;
    int gpu_device = 0;

    if (argc >= 2) loop_count = atoi(argv[1]);
    if (argc >= 3) num_threads = atoi(argv[2]);
    if (argc >= 4) gpu_device = atoi(argv[3]);

    g_loop_count = loop_count;

    printf("loop_count = %d, num_threads = %d, gpu_device = %d\n", loop_count, num_threads, gpu_device);
    fflush(stdout);

#if NCNN_VULKAN
    printf("Creating GPU instance...\n");
    fflush(stdout);
    ncnn::create_gpu_instance();
    printf("Getting GPU device %d...\n", gpu_device);
    fflush(stdout);
    g_vkdev = ncnn::get_gpu_device(gpu_device);
    if (!g_vkdev)
    {
        fprintf(stderr, "Failed to get GPU device %d\n", gpu_device);
        return -1;
    }

    g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
    g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);

    // === GPU Default (pack=1, unoptimized) ===
    ncnn::Option opt_default;
    opt_default.num_threads = num_threads;
    opt_default.use_vulkan_compute = true;
    opt_default.use_packing_layout = false;  // forces pack=1 (baseline shaders)
    opt_default.blob_vkallocator = g_blob_vkallocator;
    opt_default.workspace_vkallocator = g_blob_vkallocator;
    opt_default.staging_vkallocator = g_staging_vkallocator;

    printf("\n=== GPU Default (pack=1, baseline shaders) ===\n");
    benchmark_mha("GPU-Default", 64, 2, 16, opt_default);
    benchmark_mha("GPU-Default", 64, 2, 64, opt_default);
    benchmark_mha("GPU-Default", 64, 4, 16, opt_default);
    benchmark_mha("GPU-Default", 64, 4, 64, opt_default);
    benchmark_mha("GPU-Default", 128, 4, 16, opt_default);
    benchmark_mha("GPU-Default", 128, 4, 64, opt_default);
    benchmark_mha("GPU-Default", 128, 8, 16, opt_default);
    benchmark_mha("GPU-Default", 128, 8, 64, opt_default);
    benchmark_mha("GPU-Default", 256, 8, 16, opt_default);
    benchmark_mha("GPU-Default", 256, 8, 64, opt_default);
    benchmark_mha("GPU-Default", 512, 8, 16, opt_default);
    benchmark_mha("GPU-Default", 512, 8, 64, opt_default);

    // === GPU Optimized (pack4 + P2 larger shared tile) ===
    ncnn::Option opt_optimized;
    opt_optimized.num_threads = num_threads;
    opt_optimized.use_vulkan_compute = true;
    opt_optimized.use_packing_layout = true;  // enables pack4 with P2 optimization
    opt_optimized.blob_vkallocator = g_blob_vkallocator;
    opt_optimized.workspace_vkallocator = g_blob_vkallocator;
    opt_optimized.staging_vkallocator = g_staging_vkallocator;

    printf("\n=== GPU Optimized (pack4 + P2 larger tile 16x16) ===\n");
    benchmark_mha("GPU-Optimized", 64, 2, 16, opt_optimized);
    benchmark_mha("GPU-Optimized", 64, 2, 64, opt_optimized);
    benchmark_mha("GPU-Optimized", 64, 4, 16, opt_optimized);
    benchmark_mha("GPU-Optimized", 64, 4, 64, opt_optimized);
    benchmark_mha("GPU-Optimized", 128, 4, 16, opt_optimized);
    benchmark_mha("GPU-Optimized", 128, 4, 64, opt_optimized);
    benchmark_mha("GPU-Optimized", 128, 8, 16, opt_optimized);
    benchmark_mha("GPU-Optimized", 128, 8, 64, opt_optimized);
    benchmark_mha("GPU-Optimized", 256, 8, 16, opt_optimized);
    benchmark_mha("GPU-Optimized", 256, 8, 64, opt_optimized);
    benchmark_mha("GPU-Optimized", 512, 8, 16, opt_optimized);
    benchmark_mha("GPU-Optimized", 512, 8, 64, opt_optimized);

    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
    ncnn::destroy_gpu_instance();
#else
    fprintf(stderr, "Vulkan not enabled\n");
    return -1;
#endif

    printf("\nDone.\n");
    return 0;
}