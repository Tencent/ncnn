// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

#include "benchmark.h"
#include "layer_type.h"
#include "cpu.h"
#include "net.h"
#include "datareader.h"

#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int permute_max_elempack()
{
#if NCNN_AVX512
    if (ncnn::cpu_support_x86_avx512())
        return 16;
#endif
#if NCNN_AVX
    if (ncnn::cpu_support_x86_avx())
        return 8;
#endif
    return 4;
}

static void perf_permute(const ncnn::Mat& a, int order_type)
{
    ncnn::ParamDict pd;
    pd.set(0, order_type);
    std::vector<ncnn::Mat> weights(0);
    perf_layer("Permute", pd, weights, a, "order=%d", order_type);
}

// Keep the requested input pack: perf_layer's automatic input packing would
// otherwise replace a smaller input pack before the timed forward calls.
static int perf_permute_packing(int dims, int w, int elempack, int bits, int order_type, int threads, int h = 64, bool packing = true, int d = 1, int c = 4, int buffers = 1)
{
    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Permute);
    if (!op)
        return -1;
    if (!op->support_any_packing)
    {
        delete op;
        return 0;
    }

    ncnn::Option opt;
    opt.num_threads = threads;
    opt.use_packing_layout = packing;
    ncnn::ParamDict pd;
    pd.set(0, order_type);
    if (op->load_param(pd) != 0 || op->create_pipeline(opt) != 0)
    {
        delete op;
        return -1;
    }
    std::vector<ncnn::Mat> inputs(buffers);
    std::vector<ncnn::Mat> outputs(buffers);
    for (int i = 0; i < buffers; i++)
    {
        ncnn::Mat& a = inputs[i];
        if (dims == 2)
            a.create(w, h, (size_t)(bits / 8 * elempack), elempack);
        if (dims == 3)
            a.create(w, h, c, (size_t)(bits / 8 * elempack), elempack);
        if (dims == 4)
            a.create(w, h, d, c, (size_t)(bits / 8 * elempack), elempack);
        if (a.empty())
        {
            op->destroy_pipeline(opt);
            delete op;
            return -1;
        }
        memset(a.data, 0, a.total() * a.elemsize);
        if (op->forward(a, outputs[i], opt) != 0)
        {
            op->destroy_pipeline(opt);
            delete op;
            return -1;
        }
    }
    const size_t bytes = inputs[0].total() * inputs[0].elemsize;
    const int iterations = std::max(buffers, (int)std::max((size_t)4, std::min((size_t)128, 16 * 1024 * 1024 / bytes)));
    double times[7];
    for (int trial = 0; trial < 7; trial++)
    {
        const double start = ncnn::get_current_time();
        for (int i = 0; i < iterations; i++)
        {
            const int b = i % buffers;
            if (op->forward(inputs[b], outputs[b], opt) != 0)
            {
                op->destroy_pipeline(opt);
                delete op;
                return -1;
            }
        }
        times[trial] = (ncnn::get_current_time() - start) / iterations;
    }
    std::sort(times, times + 7);
    fprintf(stderr, "Permute dims=%d w=%d h=%d d=%d c=%d bits=%d pack=%d->%d order=%d threads=%d buffers=%d median=%.6f min=%.6f max=%.6f ms\n", dims, w, h, d, c, bits, elempack, outputs[0].elempack, order_type, threads, buffers, times[3], times[0], times[6]);
    op->destroy_pipeline(opt);
    delete op;
    return 0;
}

#if NCNN_STRING
static int perf_permute_net(int threads)
{
    ncnn::Net net;
    net.opt.num_threads = threads;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_bf16_storage = false;
    const char* param = "7767517\n2 2\nInput input 0 1 in\nPermute permute 1 1 in out 0=3\n";
    const unsigned char weights[4] = {0};
    const unsigned char* weights_ptr = weights;
    ncnn::DataReaderFromMemory mb(weights_ptr);
    if (net.load_param_mem(param) != 0 || net.load_model(mb) != 0)
        return -1;
    ncnn::Mat a = PerfMat(80, 1600, 32);
    double start = 0;
    for (int i = 0; i < 36; i++)
    {
        if (i == 4) start = ncnn::get_current_time();
        ncnn::Extractor ex = net.create_extractor();
        ncnn::Mat out;
        if (ex.input("in", a) != 0 || ex.extract("out", out, 1) != 0)
            return -1;
    }
    fprintf(stderr, "Permute Net (80,1600,32) order=3 threads=%d %.4f ms\n", threads, (ncnn::get_current_time() - start) / 32);
    return 0;
}
#endif // NCNN_STRING

int main(int argc, char** argv)
{
    if (argc > 1 && strcmp(argv[1], "--case") == 0)
    {
        if (argc != 12 && argc != 13)
        {
            fprintf(stderr, "usage: %s --case dims w h d c pack bits order threads packing [buffers]\n", argv[0]);
            return -1;
        }
        const int dims = atoi(argv[2]);
        const int w = atoi(argv[3]);
        const int h = atoi(argv[4]);
        const int d = atoi(argv[5]);
        const int c = atoi(argv[6]);
        const int pack = atoi(argv[7]);
        const int bits = atoi(argv[8]);
        const int order = atoi(argv[9]);
        const int threads = atoi(argv[10]);
        const int packing = atoi(argv[11]);
        const int buffers = argc == 13 ? atoi(argv[12]) : 1;
        if (dims < 2 || dims > 4 || w <= 0 || h <= 0 || d <= 0 || c <= 0 || threads <= 0 || buffers <= 0
            || (pack != 1 && pack != 4 && pack != 8 && pack != 16) || pack > permute_max_elempack()
            || (bits != 16 && bits != 32) || order < 0 || order >= (dims == 2 ? 2 : dims == 3 ? 6 : 24)
            || (packing != 0 && packing != 1))
            return -1;
        return perf_permute_packing(dims, w, pack, bits, order, threads, h, packing != 0, d, c, buffers);
    }
    if (argc > 1 && strcmp(argv[1], "--spatial") == 0)
    {
        const int sizes[] = {7, 32, 64, 257, 1024};
        for (int bits = 16; bits <= 32; bits *= 2)
        {
            for (int pack = 4; pack <= permute_max_elempack(); pack *= 2)
            {
                for (int i = 0; i < 5; i++)
                {
                    for (int packing = 0; packing < 2; packing++)
                    {
                        if (perf_permute_packing(3, sizes[i], pack, bits, 1, 1, sizes[i], packing != 0, 1, 1) != 0)
                            return -1;
                    }
                }
            }
        }
        return 0;
    }
    const bool packed_only = argc > 1 && strcmp(argv[1], "--packed") == 0;
    if (!packed_only)
    {
        perf_permute(PerfMat(1024, 1024), 1);
        for (int order = 0; order < 6; order++)
        {
            perf_permute(PerfMat(256, 256, 32), order);
            perf_permute(PerfMat(80, 1600, 32), order);
        }
        for (int order = 0; order < 24; order++)
            perf_permute(PerfMat(19, 19, 24, 16), order);
    }
    const int packs[] = {1, 4, 8, 16};
    for (int threads = 1; threads <= 4; threads *= 4)
    {
        for (int bits = 16; bits <= 32; bits *= 2)
        {
            for (int p = 0; p < 4 && packs[p] <= permute_max_elempack(); p++)
            {
                const int widths[] = {12, 24, 32};
                for (int i = 0; i < 3; i++)
                {
                    if (perf_permute_packing(2, widths[i], packs[p], bits, 1, threads) != 0)
                        return -1;
                }
                for (int order = 0; order < 24; order++)
                {
                    if (perf_permute_packing(4, 16, packs[p], bits, order, threads, 8, true, 4, 1) != 0)
                        return -1;
                }
                for (int order = 0; order < 6; order++)
                {
                    if (perf_permute_packing(3, 64, packs[p], bits, order, threads) != 0)
                        return -1;
                }
            }
        }
#if NCNN_STRING
        if (perf_permute_net(threads) != 0)
            return -1;
#endif // NCNN_STRING
    }
    return 0;
}
