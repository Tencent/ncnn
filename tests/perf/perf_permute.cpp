// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

#include "benchmark.h"
#include "layer_type.h"
#include "cpu.h"
#include "net.h"
#include "datareader.h"

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
static int perf_permute_packing(int elempack, int bits, int order_type, int threads)
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
    ncnn::ParamDict pd;
    pd.set(0, order_type);
    op->load_param(pd);
    op->create_pipeline(opt);
    ncnn::Mat a(64, 64, 4, (size_t)(bits / 8 * elempack), elempack);
    memset(a.data, 0, a.total() * a.elemsize);
    ncnn::Mat out;
    for (int i = 0; i < 4; i++)
    {
        if (op->forward(a, out, opt) != 0)
        {
            op->destroy_pipeline(opt);
            delete op;
            return -1;
        }
    }
    const double start = ncnn::get_current_time();
    for (int i = 0; i < 32; i++)
        op->forward(a, out, opt);
    const double elapsed = (ncnn::get_current_time() - start) / 32;
    fprintf(stderr, "Permute (64,64,%d) bits=%d pack=%d->%d order=%d threads=%d %.4f ms\n", 4 * elempack, bits, elempack, out.elempack, order_type, threads, elapsed);
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
                for (int order = 0; order < 6; order++)
                {
                    if (perf_permute_packing(packs[p], bits, order, threads) != 0)
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
