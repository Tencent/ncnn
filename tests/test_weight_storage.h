// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEST_WEIGHT_STORAGE_H
#define TEST_WEIGHT_STORAGE_H

#include "command.h"
#include "datareader.h"
#include "gpu.h"
#include "net.h"
#include "testutil.h"

#include <errno.h>

// General disk-weight cases use a private writable directory, not system TMPDIR.
// Default-directory fallback is exercised separately by the load-failure test.
static const char* g_weight_storage_test_directory = 0;

class WeightStorageTestDirectory
{
public:
    WeightStorageTestDirectory()
    {
        path[0] = 0;
    }
    ~WeightStorageTestDirectory()
    {
        if (remove() != 0)
            fprintf(stderr, "weight storage temporary directory cleanup failed: %s\n", path);
    }

    int create()
    {
#if NCNN_STDIO && (!defined(__ANDROID__) || __ANDROID_API__ >= 21) && ((defined(_WIN32) && !NCNN_WINXP) || defined(__ANDROID__) || defined(__OHOS__) || defined(__linux__) || __APPLE__)
#if defined(_WIN32)
        const unsigned long process = GetCurrentProcessId();
#else
        const unsigned long process = (unsigned long)getpid();
#endif
        for (int i = 0; i < 64; i++)
        {
            snprintf(path, sizeof(path), "./ncnn-weight-storage-test-%lu-%d", process, i);
#if defined(_WIN32)
            if (CreateDirectoryA(path, 0))
#else
            if (mkdir(path, 0700) == 0)
#endif
            {
                g_weight_storage_test_directory = path;
                return 0;
            }
#if defined(_WIN32)
            if (GetLastError() != ERROR_ALREADY_EXISTS)
#else
            if (errno != EEXIST)
#endif
                break;
        }
        path[0] = 0;
        return -1;
#else
        // Platforms without temporary-file support retain the fallback tests.
        return 0;
#endif
    }

    int remove()
    {
        if (!path[0])
            return 0;
#if defined(_WIN32)
        if (!RemoveDirectoryA(path))
            return -1;
#elif defined(__ANDROID__) || defined(__OHOS__) || defined(__linux__) || __APPLE__
        if (rmdir(path) != 0)
            return -1;
#endif
        path[0] = 0;
        g_weight_storage_test_directory = 0;
        return 0;
    }

private:
    WeightStorageTestDirectory(const WeightStorageTestDirectory&);
    WeightStorageTestDirectory& operator=(const WeightStorageTestDirectory&);
    char path[128];
};

static const char* weight_storage_test_directory()
{
    return g_weight_storage_test_directory;
}

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

// Record only: preparing resident views must not submit unrelated queued uploads.
static int record_weight_storage_resident_tier(ncnn::VkWeightAllocator& weights, ncnn::VkTransfer& upload, int storage_type, const ncnn::Option& opt, bool force_staging = false)
{
    std::vector<ncnn::Mat> sources;
    std::vector<ncnn::VkMat> destinations;
    int ret = weights.get_resident_uploads(storage_type, sources, destinations);
    for (size_t i = 0; ret == 0 && i < sources.size(); i++)
    {
        // Exercise bounded staging even on a unified-memory Vulkan device.
        if (force_staging)
            destinations[i].data->mapped_ptr = 0;
        const ncnn::VkMat& destination = destinations[i];
        upload.record_upload(sources[i], destination, opt);
        ret = upload.error();
    }
    return ret;
}

static int upload_weight_storage_residents(ncnn::VkWeightAllocator& weights, ncnn::VkTransfer& upload, const ncnn::Option& opt)
{
    for (int storage_type = 0; storage_type < 2; storage_type++)
    {
        int ret = record_weight_storage_resident_tier(weights, upload, storage_type, opt);
        if (ret == 0)
            ret = upload.submit_and_wait();
        weights.finish_resident_uploads(storage_type, ret);
        if (ret != 0)
            return ret;
        if (upload.reset() != 0)
            return upload.error();
    }
    return 0;
}

static int load_weight_storage_net(ncnn::Net& net, const ncnn::Option& opt, float scale, int layer_count, bool distinct = false, bool use_test_directory = true)
{
    net.opt = opt;
    if (use_test_directory)
        net.set_weight_file_directory(weight_storage_test_directory());
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
