// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEST_WEIGHT_STORAGE_H
#define TEST_WEIGHT_STORAGE_H

#include "command.h"
#include "datareader.h"
#include "gpu.h"
#include "net.h"
#include "testutil.h"

static ncnn::Option weight_storage_option()
{
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_cooperative_matrix = false;
    return opt;
}

static int load_weight_storage_net(ncnn::Net& net, const ncnn::Option& opt, float scale, int layer_count, bool distinct = false)
{
    net.opt = opt;
    net.set_vulkan_device(ncnn::get_default_gpu_index());

    char line[256];
    snprintf(line, sizeof(line), "7767517\n%d %d\nInput input 0 1 b0\n", layer_count + 1, layer_count + 1);
    std::string param = line;
    std::vector<unsigned char> model;
    int width = 16;
    for (int i = 0; i < layer_count; i++)
    {
        const int next_width = distinct ? 16 + ((i + 1) % 3) * 4 : 16;
        snprintf(line, sizeof(line), "Gemm gemm%d 1 1 b%d b%d 5=1 6=1 8=%d 9=%d 10=-1\n", i, i, i + 1, next_width, width);
        param += line;
        size_t offset = model.size();
        model.resize(offset + 4 + width * next_width * 4, 0);
        for (int y = 0; y < width; y++)
        {
            for (int x = 0; x < next_width; x++)
            {
                float value = x == y ? scale : 0.f;
                if (distinct)
                {
                    // Different noncommuting matrices expose stale or reordered bindings.
                    value += ((x * 3 + y * 7 + i * 11) % 13 - 6) * 0.00390625f;
                    if (x == (y + i + 1) % next_width)
                        value += 0.125f;
                }
                memcpy(model.data() + offset + 4 + (y * next_width + x) * 4, &value, 4);
            }
        }
        width = next_width;
    }

    int ret = net.load_param_mem(param.c_str());
    if (ret != 0)
        return ret;
    const unsigned char* ptr = model.data();
    ncnn::DataReaderFromMemory reader(ptr, model.size());
    return net.load_model(reader);
}

static void configure_weight_storage_extractor(ncnn::Extractor& ex, const ncnn::Option& opt)
{
    ex.set_blob_vkallocator(opt.blob_vkallocator);
    ex.set_workspace_vkallocator(opt.blob_vkallocator);
    ex.set_staging_vkallocator(opt.staging_vkallocator);
}

static ncnn::Mat scale_weight_storage_input(const ncnn::Mat& input, float scale)
{
    ncnn::Mat expected = input.clone();
    for (size_t i = 0; i < expected.total(); i++)
        ((float*)expected)[i] *= scale;
    return expected;
}

#endif // TEST_WEIGHT_STORAGE_H
