// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "benchmark.h"
#include "command.h"
#include "../testutil.h"
#include "gpu.h"
#include "layer.h"
#include "modelbin.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace ncnn {
void begin_weight_upload(VkWeightAllocator* allocator);
int discard_weight_file_cache(const VkWeightAllocator* allocator);
} // namespace ncnn

struct WeightBenchmarkModel
{
    explicit WeightBenchmarkModel(const ncnn::Option& _opt)
        : opt(_opt)
    {
    }
    ~WeightBenchmarkModel()
    {
        for (size_t i = 0; i < layers.size(); i++)
        {
            layers[i]->destroy_pipeline(opt);
            delete layers[i];
        }
    }
    ncnn::Option opt;
    std::vector<ncnn::Layer*> layers;
};

class WeightBenchmarkAllocator : public ncnn::VkWeightAllocator
{
public:
    explicit WeightBenchmarkAllocator(const ncnn::VulkanDevice* vkdev)
        : ncnn::VkWeightAllocator(vkdev)
    {
    }
    struct Source
    {
        ncnn::VkBufferMemory* memory;
        const void* data;
        size_t size;
    };
    virtual ncnn::VkBufferMemory* fastMalloc(size_t size)
    {
        ncnn::VkBufferMemory* memory = ncnn::VkWeightAllocator::fastMalloc(size);
        if (memory)
        {
            Source source;
            source.memory = memory;
            source.data = memory->external_ptr;
            source.size = size;
            sources.push_back(source);
        }
        return memory;
    }
    virtual void fastFree(ncnn::VkBufferMemory* memory)
    {
        for (size_t i = 0; i < sources.size(); i++)
            if (sources[i].memory == memory)
                sources[i].memory = 0;
        ncnn::VkWeightAllocator::fastFree(memory);
    }
    std::vector<Source> sources;
};

static uint64_t process_read_bytes()
{
#if defined(__linux__)
    FILE* file = fopen("/proc/self/io", "r");
    if (!file)
        return 0;
    char line[256];
    unsigned long long bytes = 0;
    while (fgets(line, sizeof(line), file))
        if (sscanf(line, "read_bytes: %llu", &bytes) == 1)
            break;
    fclose(file);
    return (uint64_t)bytes;
#else
    return 0;
#endif
}

static int benchmark_weights(int gpu, int width, int rows, int count, int loops, int queue_mode, bool cold, int residency)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu);
    const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
    WeightAllocationAudit allocation_audit(vkdev);
    WeightBenchmarkAllocator weights(vkdev);
    ncnn::VkWeightStagingAllocator upload_staging(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    if (weights.set_weights_in_host_disk(residency != 3))
        return -1;
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = true;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_cooperative_matrix = false;
    opt.blob_vkallocator = &blobs;
    opt.workspace_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_host_cache_size = (size_t)width * width * 4 * 2;
    opt.weight_device_cache_size = opt.weight_host_cache_size;
    const size_t weight_bytes = (size_t)width * width * 4;
    if (residency == 1 || residency == 2)
    {
        if (weight_bytes > (size_t)-1 / ((size_t)count + 2))
            return -1;
        opt.weight_device_budget = weight_bytes * (residency == 1 ? count / 4 + 2 : count + 2);
        opt.weight_host_budget = weight_bytes * (residency == 1 ? count / 4 + 1 : count + 2);
        opt.weight_host_cache_size = weight_bytes;
        if (weights.set_weight_budgets(opt.weight_device_budget, opt.weight_host_budget))
            return -1;
    }
    WeightBenchmarkModel model(opt);
    const double load_start = ncnn::get_current_time();
    for (int i = 0; i < count; i++)
    {
        ncnn::Layer* layer = ncnn::create_layer_vulkan("Gemm");
        if (!layer)
            return -1;
        layer->vkdev = vkdev;
        model.layers.push_back(layer);
        ncnn::ParamDict pd;
        pd.set(5, 1);
        pd.set(6, 1);
        pd.set(8, width);
        pd.set(9, width);
        pd.set(10, -1);
        if (layer->load_param(pd))
            return -1;
        // generate only one source weight at a time
        ncnn::Mat source(width, width);
        if (source.empty())
            return -1;
        source.fill(0.f);
        for (int j = 0; j < width; j++)
            source.row(j)[j] = 1.f;
        ncnn::ModelBinFromMatArray reader(&source);
        if (layer->load_model(reader) || layer->create_pipeline(opt))
            return -1;
        ncnn::Option upload_opt = opt;
        upload_opt.blob_vkallocator = &weights;
        upload_opt.workspace_vkallocator = &weights;
        upload_opt.staging_vkallocator = &upload_staging;
        ncnn::VkTransfer upload(vkdev);
        ncnn::begin_weight_upload(&weights);
        if (layer->upload_model(upload, upload_opt) || upload.submit_and_wait())
            return -1;
    }
    const double finalize_start = ncnn::get_current_time();
    if (weights.finalize_weights(opt.weight_device_cache_size, opt.weight_host_cache_size))
        return -1;
    const double finalize_time = ncnn::get_current_time() - finalize_start;
    const double load_time = ncnn::get_current_time() - load_start;
    const size_t device_load_peak = allocation_audit.peak[0];
    const size_t host_load_peak = allocation_audit.peak[1];
    uint64_t prepared_bytes = 0;
    for (size_t i = 0; i < weights.sources.size(); i++)
        prepared_bytes += weights.sources[i].size;
    double cold_scan_time = 0;
    uint64_t cold_scan_bytes = 0;
    uint64_t cold_scan_physical_bytes = 0;
    if (cold && residency != 3)
    {
        // measure a sequential cold scan independently of gpu work and readahead overlap
        std::vector<unsigned char> scan(64 * 1024);
        if (ncnn::discard_weight_file_cache(&weights))
            return -1;
        const uint64_t read_start = process_read_bytes();
        const double start = ncnn::get_current_time();
        for (size_t i = 0; i < weights.sources.size(); i++)
        {
            const WeightBenchmarkAllocator::Source& source = weights.sources[i];
            if (!source.data)
                continue;
            for (size_t offset = 0; offset < source.size;)
            {
                const size_t size = std::min(scan.size(), source.size - offset);
                // keep the full copy observable even when the scan buffer is overwritten
                void* (*volatile copy)(void*, const void*, size_t) = memcpy;
                copy(scan.data(), (const unsigned char*)source.data + offset, size);
                offset += size;
            }
            cold_scan_bytes += source.size;
        }
        cold_scan_time = ncnn::get_current_time() - start;
        cold_scan_physical_bytes = process_read_bytes() - read_start;
    }
    ncnn::Mat input(width, rows);
    input.fill(0.25f);
    reset_weight_benchmark();
    ncnn::VkCompute cmd(weights, opt, queue_mode);
    double total_time = 0;
    uint64_t physical_read_bytes = 0;
    for (int iteration = 0; iteration < loops; iteration++)
    {
        if (iteration && cmd.reset())
            return -1;
        if (cold && residency != 3 && ncnn::discard_weight_file_cache(&weights))
            return -1;
        const uint64_t read_start = process_read_bytes();
        const double start = ncnn::get_current_time();
        ncnn::VkMat current;
        cmd.record_upload(input, current, opt);
        for (int i = 0; i < count; i++)
        {
            std::vector<ncnn::VkMat> bottom(1, current);
            std::vector<ncnn::VkMat> top(1);
            if (model.layers[i]->forward(bottom, top, cmd, opt))
                return -1;
            current = top[0];
        }
        ncnn::Mat output;
        cmd.record_download(current, output, opt);
        if (cmd.submit_and_wait())
            return -1;
        const double elapsed = ncnn::get_current_time() - start;
        total_time += elapsed;
        physical_read_bytes += process_read_bytes() - read_start;
        if (output.total() * output.elempack != input.total())
            return -1;
        for (size_t i = 0; i < output.total() * output.elempack; i++)
            if (!(fabsf(((const float*)output)[i] - 0.25f) <= 0.0001f))
                return -1;
        fprintf(stderr, "iteration %d end_to_end_ms=%.3f\n", iteration, elapsed);
    }
    fprintf(stderr, "weight benchmark gpu=%s width=%d rows=%d layers=%d loops=%d queue_mode=%d cold=%d residency=%d unified=%d separate_queue=%d separate_family=%d\n",
            vkdev->info.device_name(), width, rows, count, loops, queue_mode, cold, residency, unified, (int)weight_benchmark_counts()["separate_transfer_queue"], (int)weight_benchmark_counts()["separate_transfer_family"]);
    fprintf(stderr, "prepared_bytes=%llu load_ms=%.3f finalize_ms=%.3f host_peak_bytes=%llu device_peak_bytes=%llu\n",
            (unsigned long long)prepared_bytes, load_time, finalize_time, (unsigned long long)weight_benchmark_counts()["host_allocation_size"], (unsigned long long)weight_benchmark_counts()["device_allocation_size"]);
    uint64_t storage_bytes[3] = {};
    unsigned int storage_count[3] = {};
    size_t resident_allocation[2] = {};
    std::vector<VkDeviceMemory> resident_memory;
    for (size_t i = 0; i < weights.sources.size(); i++)
    {
        const WeightBenchmarkAllocator::Source& source = weights.sources[i];
        const ncnn::VkBufferMemory* memory = source.memory;
        if (!memory)
            continue;
        storage_bytes[memory->storage_type] += source.size;
        storage_count[memory->storage_type]++;
        if (memory->storage_type < 2 && std::find(resident_memory.begin(), resident_memory.end(), memory->memory) == resident_memory.end())
        {
            VkMemoryRequirements requirements;
            ncnn::vkGetBufferMemoryRequirements(vkdev->vkdevice(), memory->buffer, &requirements);
            resident_allocation[memory->storage_type] += (size_t)requirements.size;
            resident_memory.push_back(memory->memory);
        }
    }
    fprintf(stderr, "layout_device=%u/%llu layout_host=%u/%llu layout_file=%u/%llu device_resident=%llu host_resident=%llu device_exchange=%llu host_exchange=%llu device_load_peak=%llu host_load_peak=%llu\n",
            storage_count[0], (unsigned long long)storage_bytes[0], storage_count[1], (unsigned long long)storage_bytes[1], storage_count[2], (unsigned long long)storage_bytes[2],
            (unsigned long long)resident_allocation[0], (unsigned long long)resident_allocation[1], (unsigned long long)weight_benchmark_counts()["device_allocation_size"], (unsigned long long)weight_benchmark_counts()["host_allocation_size"],
            (unsigned long long)device_load_peak, (unsigned long long)host_load_peak);
    fprintf(stderr, "load_peak_tracked=1\n");
    std::map<std::string, uint64_t>& stats = weight_benchmark_counts();
    fprintf(stderr, "upload_timing_available=%d\n", (int)stats["upload_time_available"]);
    fprintf(stderr, "host_cache_hits=%llu host_resident_uploaded=%llu upload_chunks=%llu unified_written=%llu\n",
            (unsigned long long)stats["host_cache_hits"], (unsigned long long)stats["host_resident_bytes_uploaded"], (unsigned long long)stats["upload_chunks"],
            (unsigned long long)stats["unified_bytes_written"]);
    fprintf(stderr, "cold_scan_ms=%.3f cold_scan_MB_s=%.3f cold_scan_physical_bytes=%llu\n",
            cold_scan_time, cold_scan_time ? cold_scan_bytes / (cold_scan_time * 1000.) : 0., (unsigned long long)cold_scan_physical_bytes);
    fprintf(stderr, "file_reads=%llu bytes_read=%llu host_staged=%llu device_uploaded=%llu cache_hits=%llu cache_misses=%llu transfer_submissions=%llu compute_submissions=%llu ownership_transfers=%llu\n",
            (unsigned long long)stats["file_reads"], (unsigned long long)stats["bytes_read"], (unsigned long long)stats["host_bytes_staged"], (unsigned long long)stats["bytes_uploaded"],
            (unsigned long long)stats["cache_hits"], (unsigned long long)stats["cache_misses"], (unsigned long long)stats["transfer_submissions"], (unsigned long long)stats["compute_submissions"], (unsigned long long)stats["ownership_transfers"]);
    fprintf(stderr, "read_ms=%.3f upload_gpu_ms=%.3f compute_gpu_ms=%.3f transfer_wait_ms=%.3f compute_wait_ms=%.3f inference_mean_ms=%.3f read_MB_s=%.3f upload_MB_s=%.3f physical_read_bytes=%llu pending_transfer=%u pending_compute=%u\n",
            weight_benchmark_times()["disk_read_time"], weight_benchmark_times()["upload_time"], weight_benchmark_times()["compute_time"], weight_benchmark_times()["transfer_wait_time"], weight_benchmark_times()["compute_wait_time"], total_time / loops,
            weight_benchmark_times()["disk_read_time"] ? stats["bytes_read"] / (weight_benchmark_times()["disk_read_time"] * 1000.) : 0., weight_benchmark_times()["upload_time"] ? stats["bytes_uploaded"] / (weight_benchmark_times()["upload_time"] * 1000.) : 0., (unsigned long long)physical_read_bytes, (unsigned int)stats["peak_pending_transfer"], (unsigned int)stats["peak_pending_compute"]);
    return 0;
}

int main(int argc, char** argv)
{
    if (ncnn::get_gpu_count() == 0)
        return 0;
    const int gpu = argc > 1 ? atoi(argv[1]) : ncnn::get_default_gpu_index();
    const int width = argc > 2 ? atoi(argv[2]) : 1024;
    const int rows = argc > 3 ? atoi(argv[3]) : 16;
    const int count = argc > 4 ? atoi(argv[4]) : 48;
    const int loops = argc > 5 ? atoi(argv[5]) : 3;
    const int queue_mode = argc > 6 ? atoi(argv[6]) : 0;
    if (gpu < 0 || gpu >= ncnn::get_gpu_count() || width < 16 || width > 16384 || width % 16 || rows < 1 || count < 1 || loops < 1 || queue_mode < 0 || queue_mode > 2)
        return -1;
    const bool cold = argc > 7 && atoi(argv[7]) != 0;
    // residency modes: file exchange, mixed resident, all resident, ordinary device
    const int residency = argc > 8 ? atoi(argv[8]) : 0;
    if (residency < 0 || residency > 3)
        return -1;
    const int ret = benchmark_weights(gpu, width, rows, count, loops, queue_mode, cold, residency);
    ncnn::destroy_gpu_instance();
    return ret;
}
