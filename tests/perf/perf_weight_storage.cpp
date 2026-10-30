// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

#include "benchmark.h"
#include "datareader.h"
#include "gpu.h"
#include "net.h"

#include <float.h>
#include <stdio.h>

static int perf_weight_storage(bool host, bool disk, size_t device_memory_limit, size_t host_memory_limit)
{
    const int width = 128;
    const int layers = 24;
    const int loops = 5;
    ncnn::Net net;
    net.opt.num_threads = 1;
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_cooperative_matrix = false;
    net.opt.use_weights_in_host_memory = host;
    net.opt.use_weights_in_disk = disk;
    net.opt.weight_device_memory_limit = device_memory_limit;
    net.opt.weight_host_memory_limit = host_memory_limit;

    char line[256];
    snprintf(line, sizeof(line), "7767517\n%d %d\nInput input 0 1 b0\n", layers + 1, layers + 1);
    std::string param = line;
    std::vector<unsigned char> model;
    for (int i = 0; i < layers; i++)
    {
        snprintf(line, sizeof(line), "Gemm gemm%d 1 1 b%d b%d 5=1 6=1 8=%d 9=%d 10=-1\n", i, i, i + 1, width, width);
        param += line;
        size_t offset = model.size();
        model.resize(offset + 4 + width * width * sizeof(float), 0);
        const float scale = 1.f;
        for (int j = 0; j < width; j++)
            memcpy(model.data() + offset + 4 + (j * width + j) * sizeof(float), &scale, sizeof(float));
    }

    double start = ncnn::get_current_time();
    int ret = net.load_param_mem(param.c_str());
    if (ret != 0)
        return ret;
    const unsigned char* ptr = model.data();
    ncnn::DataReaderFromMemory reader(ptr, model.size());
    ret = net.load_model(reader);
    if (ret != 0)
        return ret;
    double load_time = ncnn::get_current_time() - start;

    ncnn::Mat input = PerfMat(width, 16);
    double time_min = DBL_MAX;
    double time_max = 0.;
    double time_sum = 0.;
    for (int i = 0; i < loops + 2; i++)
    {
        start = ncnn::get_current_time();
        ncnn::Extractor ex = net.create_extractor();
        ncnn::Mat output;
        ret = ex.input("b0", input);
        if (ret != 0)
            return ret;
        ret = ex.extract("b24", output);
        if (ret != 0)
            return ret;
        double elapsed = ncnn::get_current_time() - start;
        if (i < 2)
            continue;
        time_min = std::min(time_min, elapsed);
        time_max = std::max(time_max, elapsed);
        time_sum += elapsed;
    }

    fprintf(stdout, "weight storage host=%d disk=%d width=%d layers=%d device_limit=%llu host_limit=%llu load=%.2fms min=%.2fms max=%.2fms avg=%.2fms\n",
            host, disk, width, layers, (unsigned long long)device_memory_limit, (unsigned long long)host_memory_limit,
            load_time, time_min, time_max, time_sum / loops);
    return 0;
}

int main()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    return 0
           || perf_weight_storage(false, false, 0, 0)
           || perf_weight_storage(true, false, 0, 0)
           || perf_weight_storage(false, true, 0, 0)
           || perf_weight_storage(false, true, 256 * 1024, 256 * 1024);
}
