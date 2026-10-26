// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "command.h"
#include "gpu.h"
#include "testutil.h"
#if defined(__linux__)
#include <signal.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace ncnn {
void begin_weight_upload(VkWeightAllocator* allocator);
int get_weight_source_generation(const VkAllocator* allocator, uint64_t& generation);
int get_weight_cache_limits(const VkWeightAllocator* allocator, size_t& device_budget, size_t& host_budget, size_t& device_size, size_t& host_size);
} // namespace ncnn

static int write_weight(ncnn::VkBufferMemory* memory, size_t offset, const void* data, size_t size)
{
    if (!memory || !memory->external_ptr || !(memory->access_flags & VK_ACCESS_HOST_WRITE_BIT)
            || offset > memory->external_size || size > memory->external_size - offset)
        return -1;
    memcpy((unsigned char*)memory->external_ptr + offset, data, size);
    return 0;
}

static int read_weight(const ncnn::VkBufferMemory* memory, size_t offset, void* data, size_t size)
{
    if (!memory || !memory->external_ptr || offset > memory->external_size || size > memory->external_size - offset)
        return -1;
    memcpy(data, (const unsigned char*)memory->external_ptr + offset, size);
    return 0;
}

static size_t weight_resident_size(const ncnn::VkMat* weights, int count, int storage_type)
{
    size_t size = 0;
    std::vector<VkDeviceMemory> allocations;
    for (int i = 0; i < count; i++)
    {
        const ncnn::VkBufferMemory* memory = weights[i].data;
        if (!memory || memory->storage_type != storage_type || !memory->buffer
                || std::find(allocations.begin(), allocations.end(), memory->memory) != allocations.end())
            continue;
        VkMemoryRequirements requirements;
        ncnn::vkGetBufferMemoryRequirements(weights[i].allocator->vkdev->vkdevice(), memory->buffer, &requirements);
        size += (size_t)requirements.size;
        allocations.push_back(memory->memory);
    }
    return size;
}

static int gpu_index = -1;

static int test_native_allocation_budget()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkMat output(4096, (size_t)4u, &blobs);
    if (output.empty())
        return -1;
    bool needs_upload = false;
    const WeightAllocationAudit::Failure failures[] = {
        WeightAllocationAudit::FAIL_NONE,
        WeightAllocationAudit::FAIL_ALLOCATE,
        WeightAllocationAudit::FAIL_BIND,
        WeightAllocationAudit::FAIL_MAP,
        WeightAllocationAudit::FAIL_SUBMIT
    };
    for (size_t i = 0; i < sizeof(failures) / sizeof(failures[0]); i++)
    {
        const WeightAllocationAudit::Failure failure = failures[i];
        if (failure == WeightAllocationAudit::FAIL_SUBMIT && !needs_upload)
            continue;
        WeightAllocationAudit audit(vkdev, failure);
        {
            ncnn::VkWeightAllocator store(vkdev);
            if (store.set_weights_in_host_disk(true) || store.set_weight_budgets(17408, unified ? 0 : 4096))
                return -1;
            const int sizes[] = {1024, 2048, 16384};
            ncnn::VkMat weights[3];
            for (int i = 0; i < 3; i++)
            {
                ncnn::begin_weight_upload(&store);
                weights[i].create(sizes[i] / 4, (size_t)4u, &store);
                unsigned char data[16384] = {};
                if (weights[i].empty() || write_weight(weights[i].data, 0, data, sizes[i]))
                    return -1;
            }
            const int ret = store.finalize_weights(16384, 1024);
            if ((ret != 0) != (failure != WeightAllocationAudit::FAIL_NONE) || audit.failed != (failure != WeightAllocationAudit::FAIL_NONE))
                return -1;
            if (audit.peak[0] > 17408 || audit.peak[1] > (unified ? 0 : 4096))
                return -1;

            if (failure == WeightAllocationAudit::FAIL_NONE)
            {
                needs_upload = !weights[0].data->mapped_ptr;
                ncnn::Option opt;
                opt.blob_vkallocator = &blobs;
                reset_weight_benchmark();
                ncnn::VkCompute cmd(store, opt);
                cmd.record_clone(weights[2], output, opt);
                if (cmd.submit_and_wait() || audit.current[0] > 17408 || audit.current[1] > (unified ? 0 : 4096))
                    return -1;
            }
            else if (store.finalize_weights() == 0 || store.fastMalloc(16))
                return -1;
        }
        if (audit.current[0] || audit.current[1])
            return -1;
    }
    return 0;
}

static int test_prepared_bytes(int mode, int elempack, bool flatten, int batch, bool host)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator disk(vkdev);
    ncnn::VkWeightAllocator resident(vkdev, host);
    ncnn::VkWeightStagingAllocator staging(vkdev);
    if (disk.set_weights_in_host_disk(true))
        return -1;

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = mode == 1;
    opt.use_bf16_storage = mode == 2;
    opt.blob_vkallocator = &disk;
    opt.staging_vkallocator = &staging;

    ncnn::Mat src(7, 3, 2, 2, (size_t)4u * elempack, elempack, batch);
    for (int b = 0; b < batch; b++)
    {
        ncnn::Mat m = src.batch(b);
        float* p = m;
        for (size_t i = 0; i < m.total() * elempack; i++)
            p[i] = (float)((int)(i % 53) - 26) * 0.125f + b;
    }

    ncnn::VkMat file_weight;
    ncnn::VkTransfer upload_file(vkdev);
    upload_file.record_upload(src, file_weight, opt, flatten);
    if (upload_file.error() || file_weight.empty() || upload_file.pending_upload_total() != 0)
        return -1;
    if (file_weight.data->storage_type != 2 || file_weight.buffer() || file_weight.data->memory || file_weight.mapped_ptr())
        return -1;
    if (upload_file.submit_and_wait() || upload_file.reset() || upload_file.submit_and_wait() || disk.finalize_weights())
        return -1;

    opt.blob_vkallocator = &resident;
    ncnn::VkMat resident_weight;
    ncnn::VkTransfer upload_resident(vkdev);
    upload_resident.record_upload(src, resident_weight, opt, flatten);
    if (upload_resident.submit_and_wait())
        return -1;

    // download without another precision conversion
    ncnn::Option download_opt;
    download_opt.blob_vkallocator = &staging;
    download_opt.staging_vkallocator = &staging;
    ncnn::Mat downloaded;
    reset_weight_benchmark();
    ncnn::VkCompute download(resident, download_opt);
    download.record_clone(resident_weight, downloaded, download_opt);
    if (download.submit_and_wait())
        return -1;

    if (file_weight.data->external_size > file_weight.data->capacity || file_weight.data->external_size < file_weight.buffer_capacity())
        return -1;
    std::vector<unsigned char> bytes(file_weight.data->external_size);
    if (read_weight(file_weight.data, 0, bytes.data(), bytes.size()))
        return -1;

    for (int b = 0; b < batch; b++)
    {
        const ncnn::Mat expected = downloaded.batch(b);
        const ncnn::VkMat view = file_weight.batch(b);
        const unsigned char* actual = bytes.data() + view.buffer_offset();
        for (int q = 0; q < expected.c; q++)
        {
            // channel padding in the existing cast path is unspecified
            const size_t size = (size_t)expected.w * expected.h * expected.d * expected.elemsize;
            const size_t offset = q * expected.cstep * expected.elemsize;
            if (memcmp(actual + offset, (const unsigned char*)expected.data + offset, size) != 0)
            {
                fprintf(stderr, "prepared byte mismatch mode=%d pack=%d flatten=%d batch=%d channel=%d\n", mode, elempack, flatten, b, q);
                return -1;
            }
        }
    }

    // the source lifetime ends before a later read of the same prepared range
    src.release();
    std::vector<unsigned char> again(bytes.size());
    return read_weight(file_weight.data, 0, again.data(), again.size()) || memcmp(bytes.data(), again.data(), bytes.size());
}

struct WeightReadTask
{
    const ncnn::VkBufferMemory* memory;
    unsigned char expected;
    int result;
};

static void* read_weight_thread(void* args)
{
    WeightReadTask* task = (WeightReadTask*)args;
    task->result = 0;
    for (int i = 0; i < 1000; i++)
    {
        unsigned char bytes[16];
        if (read_weight(task->memory, 0, bytes, sizeof(bytes)))
        {
            task->result = -1;
            break;
        }
        for (int j = 0; j < 16; j++)
            if (bytes[j] != task->expected)
                task->result = -1;
    }
    return 0;
}

static int test_concurrent_reads()
{
    ncnn::VkWeightAllocator allocator(ncnn::get_gpu_device(gpu_index));
    if (allocator.set_weights_in_host_disk(true))
        return -1;
    ncnn::VkBufferMemory* weights[2];
    WeightReadTask tasks[2];
    for (int i = 0; i < 2; i++)
    {
        weights[i] = allocator.fastMalloc(16);
        if (!weights[i])
            return -1;
        unsigned char bytes[16];
        memset(bytes, 42 + i, sizeof(bytes));
        if (write_weight(weights[i], 0, bytes, sizeof(bytes)))
            return -1;
        tasks[i].memory = weights[i];
        tasks[i].expected = 42 + i;
        tasks[i].result = -1;
    }
    if (allocator.finalize_weights())
        return -1;
#if NCNN_THREADS
    ncnn::Thread thread0(read_weight_thread, &tasks[0]);
    ncnn::Thread thread1(read_weight_thread, &tasks[1]);
    thread0.join();
    thread1.join();
#else
    read_weight_thread(&tasks[0]);
    read_weight_thread(&tasks[1]);
#endif
    allocator.fastFree(weights[0]);
    allocator.fastFree(weights[1]);
    return tasks[0].result || tasks[1].result;
}

static int test_external_sources()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator allocator(vkdev);
    if (allocator.set_weights_in_host_disk(true))
        return -1;
    ncnn::VkBufferMemory* first = allocator.fastMalloc(17);
    if (!first || !first->external_ptr || first->external_size != 17)
        return -1;
    void* first_address = first->external_ptr;
    memset(first_address, 7, 17);
    // later allocations extend the file without moving an existing view
    ncnn::VkBufferMemory* large = allocator.fastMalloc((size_t)0x80000000u + 4096);
    ncnn::VkBufferMemory* last = allocator.fastMalloc(16);
    if (!large || !last || first->external_ptr != first_address)
        return -1;
    for (int i = 0; i < 17; i++)
        if (((const unsigned char*)first->external_ptr)[i] != 7)
            return -1;
    const unsigned char src[16] = {4, 9, 2, 1, 3, 7, 0, 5, 6, 8, 10, 11, 12, 13, 14, 15};
    unsigned char dst[16];
    if (write_weight(last, 0, src, sizeof(src)) || allocator.finalize_weights()
            || read_weight(last, 0, dst, sizeof(dst)) || memcmp(src, dst, sizeof(src)))
        return -1;
    allocator.fastFree(first);
    allocator.fastFree(large);
    allocator.fastFree(last);
    allocator.clear();
    ncnn::VkBufferMemory* next = allocator.fastMalloc(4);
    if (!next || !next->external_ptr || next->external_size != 4)
        return -1;
    allocator.fastFree(next);
    ncnn::VkWeightAllocator ordinary(vkdev);
    ncnn::VkBufferMemory* buffer = ordinary.fastMalloc(16);
    size_t device_budget, host_budget, device_size, host_size;
    if (!buffer || buffer->external_ptr || buffer->external_size || ncnn::get_weight_cache_limits(&ordinary, device_budget, host_budget, device_size, host_size))
        return -1;
    if (ordinary.set_weight_budgets(4096, 4096) == 0)
        return -1;
    ordinary.fastFree(buffer);
    ordinary.clear();
    return ordinary.set_weight_budgets(4096, 4096);
}

class FailingWeightAllocator : public ncnn::VkWeightAllocator
{
public:
    explicit FailingWeightAllocator(const ncnn::VulkanDevice* vkdev)
        : ncnn::VkWeightAllocator(vkdev)
    {
    }

    virtual ncnn::VkBufferMemory* fastMalloc(size_t)
    {
        return 0;
    }
};

static int test_upload_error()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    FailingWeightAllocator allocator(vkdev);
    ncnn::VkWeightStagingAllocator staging(vkdev);
    ncnn::Option opt;
    opt.blob_vkallocator = &allocator;
    opt.staging_vkallocator = &staging;
    opt.use_fp16_storage = false;
    opt.use_fp16_packed = false;
    ncnn::Mat src(16);
    src.fill(1.f);
    ncnn::VkMat dst;
    ncnn::VkTransfer cmd(vkdev);
    cmd.record_upload(src, dst, opt);
    if (cmd.error() != -100 || cmd.submit_and_wait() != -100 || cmd.reset() != -100 || cmd.pending_upload_total() != 0)
        return -1;
    return 0;
}

#if defined(__linux__)
static int test_file_creation_fallback()
{
    const char* previous = getenv("TMPDIR");
    const std::string previous_value = previous ? previous : "";
    if (setenv("TMPDIR", "/dev/null", 1))
        return -1;
    ncnn::VkWeightAllocator allocator(ncnn::get_gpu_device(gpu_index), true);
    const int result = allocator.set_weights_in_host_disk(true);
    if (previous)
        setenv("TMPDIR", previous_value.c_str(), 1);
    else
        unsetenv("TMPDIR");
    if (!result)
        return -1;
    ncnn::VkBufferMemory* memory = allocator.fastMalloc(16);
    if (!memory)
        return -1;
    const bool valid = memory->storage_type != 2;
    allocator.fastFree(memory);
    return valid ? 0 : -1;
}

static int test_file_directory()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    char path[] = "/tmp/ncnn-weight-directory-XXXXXX";
    if (!mkdtemp(path))
        return -1;
    ncnn::VkWeightAllocator first(vkdev);
    ncnn::VkWeightAllocator second(vkdev);
    int ret = first.set_weight_file_directory(path) || second.set_weight_file_directory("/dev/null/ncnn-weights");
    // The caller can release or change its string after configuration.
    std::string directory = path;
    path[0] = 0;
    if (!ret)
        ret = first.set_weights_in_host_disk(true);
    // An open backing file is already unlinked, and does not block removing its directory.
    if (rmdir(directory.c_str()))
        ret = -1;
    if (!second.set_weights_in_host_disk(true) || !first.set_weight_file_directory("/tmp"))
        ret = -1;
    ncnn::VkMat weights(1024, (size_t)4u, &first);
    unsigned char data[4096] = {};
    if (weights.empty() || write_weight(weights.data, 0, data, sizeof(data)) || first.finalize_weights())
        ret = -1;
    weights.release();
    first.clear();
    if (first.set_weight_file_directory(0) || first.set_weights_in_host_disk(true))
        ret = -1;
    return ret;
}

static int test_file_write_failure()
{
    // keep the process-wide file limit scoped to this synchronous upload
    struct rlimit previous;
    if (getrlimit(RLIMIT_FSIZE, &previous) != 0)
        return -1;
    struct rlimit limited = previous;
    limited.rlim_cur = 0;
    void (*previous_signal)(int) = signal(SIGXFSZ, SIG_IGN);
    if (setrlimit(RLIMIT_FSIZE, &limited) != 0)
    {
        signal(SIGXFSZ, previous_signal);
        return -1;
    }
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    int result = 0;
    {
        ncnn::VkWeightAllocator allocator(vkdev);
        ncnn::VkWeightStagingAllocator staging(vkdev);
        if (allocator.set_weights_in_host_disk(true))
            result = -1;
        ncnn::Option opt;
        opt.use_fp16_storage = false;
        opt.use_fp16_packed = false;
        opt.blob_vkallocator = &allocator;
        opt.staging_vkallocator = &staging;
        ncnn::Mat source(16384);
        source.fill(1.f);
        ncnn::VkMat destination;
        ncnn::VkTransfer upload(vkdev);
        upload.record_upload(source, destination, opt);
        if (!upload.error() || !upload.submit_and_wait() || !allocator.finalize_weights())
            result = -1;
    }
    if (setrlimit(RLIMIT_FSIZE, &previous) != 0)
        result = -1;
    signal(SIGXFSZ, previous_signal);
    return result;
}
#endif

static int test_resident_layout(bool prefer_host)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator store(vkdev, prefer_host);
    const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
    const size_t unit = 128 * 1024;
    const size_t device_budget = 36 * unit;
    const size_t host_budget = unified ? 0 : 17 * unit;
    if (store.set_weights_in_host_disk(true) || store.set_weight_budgets(device_budget, host_budget))
        return -1;
    ncnn::VkBufferMemory* discarded = store.fastMalloc(16);
    if (!discarded)
        return -1;
    store.fastFree(discarded);
    const int sizes[6] = {2, 2, 8, 8, 32, 32};
    ncnn::VkMat weights[6];
    ncnn::VkBufferMemory* memories[6];
    int refcounts[6];
    for (int i = 0; i < 6; i++)
    {
        ncnn::begin_weight_upload(&store);
        const size_t size = sizes[i] * unit;
        weights[i].create((int)(size / 4), (size_t)4u, &store);
        if (weights[i].empty())
            return -1;
        memories[i] = weights[i].data;
        refcounts[i] = memories[i]->refcount;
        unsigned char bytes[4096];
        memset(bytes, i + 1, sizeof(bytes));
        for (size_t offset = 0; offset < size; offset += sizeof(bytes))
            if (write_weight(weights[i].data, offset, bytes, sizeof(bytes)))
                return -1;
    }
    ncnn::VkMat view(128, weights[0].data, (size_t)4u, &store);
#if NCNN_BATCH
    view.offset = 64 * sizeof(float);
#endif
    const size_t view_offset = view.buffer_offset();
    if (store.finalize_weights(32 * unit, unit) || store.finalize_weights())
        return -1;
    if (view.buffer_offset() != memories[0]->offset + view_offset)
        return -1;
    // mixed backing properties are independent of the allocator-wide flag
    store.mappable = prefer_host;
    for (int i = 0; i < 6; i++)
    {
        const int expected = i < 2 ? 0 : (!unified && i < 4 ? 1 : 2);
        if (weights[i].data != memories[i]
                || memories[i]->refcount != refcounts[i] || memories[i]->storage_type != expected)
        {
            fprintf(stderr, "resident layout mismatch weight=%d type=%d expected=%d\n", i, memories[i]->storage_type, expected);
            return -1;
        }
        if (memories[i]->external_ptr && ((const unsigned char*)memories[i]->external_ptr)[0] != i + 1)
            return -1;
    }
    if (!weight_resident_size(weights, 6, 0)
            || weight_resident_size(weights, 6, 0) > 4 * unit
            || weight_resident_size(weights, 6, 1) > (unified ? 0 : 16 * unit))
        return -1;

    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    reset_weight_benchmark();
    ncnn::VkCompute cmd(store, opt);
    ncnn::Mat view_result;
    cmd.record_clone(view, view_result, opt);
    if (cmd.submit_and_wait())
        return -1;
    for (size_t i = 0; i < view_result.total() * view_result.elemsize; i++)
        if (((const unsigned char*)view_result.data)[i] != 1)
            return -1;
#if NCNN_BENCHMARK
    if (weight_benchmark_counts()["device_allocation_size"] || weight_benchmark_counts()["host_allocation_size"])
        return -1;
#endif
    if (cmd.reset())
        return -1;
    ncnn::Mat output[6];
    for (int i = 0; i < 6; i++)
        cmd.record_clone(weights[i], output[i], opt);
    if (cmd.submit_and_wait())
        return -1;
    for (int i = 0; i < 6; i++)
        for (size_t j = 0; j < output[i].total() * output[i].elemsize; j++)
            if (((const unsigned char*)output[i].data)[j] != i + 1)
                return -1;
#if NCNN_BENCHMARK
    std::map<std::string, uint64_t>& stats = weight_benchmark_counts();
    if (weight_benchmark_counts()["device_allocation_size"] + weight_resident_size(weights, 6, 0) > device_budget
            || weight_benchmark_counts()["host_allocation_size"] + weight_resident_size(weights, 6, 1) > host_budget)
        return -1;
    if (unified)
    {
        if (weight_benchmark_counts()["host_allocation_size"] || stats["bytes_uploaded"] || stats["unified_bytes_written"] != 80 * unit)
            return -1;
    }
    else if (stats["host_resident_bytes_uploaded"] != 16 * unit || stats["bytes_read"] != 64 * unit || stats["upload_chunks"] <= 4)
        return -1;
#endif
    fprintf(stderr, "resident layout and exchange verified unified=%d device=%lu host=%lu\n", unified,
            (unsigned long)weight_resident_size(weights, 6, 0), (unsigned long)weight_resident_size(weights, 6, 1));
    return 0;
}

static int test_host_cache()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator store(vkdev);
    if (store.set_weights_in_host_disk(true))
        return -1;
    ncnn::VkMat weights[2];
    for (int i = 0; i < 2; i++)
    {
        weights[i].create(256, (size_t)4u, &store);
        unsigned char data[1024];
        memset(data, i + 1, sizeof(data));
        if (weights[i].empty() || write_weight(weights[i].data, 0, data, sizeof(data)))
            return -1;
    }
    if (store.finalize_weights())
        return -1;
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_device_cache_size = 1024;
    opt.weight_host_cache_size = 2048;
    reset_weight_benchmark();
    ncnn::VkCompute cmd(store, opt);
    const int sequence[] = {0, 1, 0, 1, 1};
    for (int i = 0; i < 5; i++)
    {
        if (i && cmd.reset())
            return -1;
        ncnn::Mat output;
        cmd.record_clone(weights[sequence[i]], output, opt);
        if (cmd.submit_and_wait())
            return -1;
        for (size_t j = 0; j < output.total() * output.elemsize; j++)
            if (((const unsigned char*)output.data)[j] != sequence[i] + 1)
                return -1;
    }
#if NCNN_BENCHMARK
    std::map<std::string, uint64_t>& stats = weight_benchmark_counts();
    if (!stats["cache_hits"])
        return -1;
    if (vkdev->info.type() == 1 || vkdev->info.type() == 3)
    {
        if (stats["file_reads"] != 4 || stats["host_cache_hits"] || stats["bytes_uploaded"] || weight_benchmark_counts()["host_allocation_size"])
            return -1;
    }
    else if (stats["file_reads"] != 2 || stats["host_cache_hits"] != 2 || stats["bytes_uploaded"] != 4096)
        return -1;
#endif
    return 0;
}

static int test_budget_boundaries()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator store(vkdev);
    const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
    if (store.set_weight_budgets(4096, 0) != (unified ? 0 : -1))
        return -1;
    size_t remaining_device, remaining_host, preferred_device, preferred_host;
    if (store.set_weights_in_host_disk(true) || store.set_weight_budgets(4096, 4096) || store.finalize_weights()
            || ncnn::get_weight_cache_limits(&store, remaining_device, remaining_host, preferred_device, preferred_host))
        return -1;
    store.clear();
    if (store.set_weight_budgets(4096, 4096))
        return -1;
    if (store.fastMalloc((size_t)-1) || store.fastMalloc(4, 4, 4, (size_t)4u, 1))
        return -1;
    for (int mode = 0; mode < 6; mode++)
    {
        store.clear();
        const size_t device_budget = mode == 0 ? 1 : mode == 1 ? 4095 : 4096;
        const size_t device_cache = mode >= 2 && mode <= 4 ? (mode == 2 ? 8192 : mode == 3 ? 4096 : 2048)
                                    : 0;
        const size_t host_cache = mode == 3 ? 8192 : 0;
        if (store.set_weight_budgets(device_budget, 4096))
            return -1;
        ncnn::VkMat weights[2];
        for (int i = 0; i < 2; i++)
        {
            ncnn::begin_weight_upload(&store);
            weights[i].create(mode >= 2 && mode <= 4 ? 1024 : 512, (size_t)4u, &store);
            unsigned char bytes[4096] = {};
            if (weights[i].empty() || write_weight(weights[i].data, 0, bytes, weights[i].total() * weights[i].elemsize))
                return -1;
        }
        const int ret = store.finalize_weights(device_cache, host_cache);
        const bool success = mode == 1 || mode == 5 || (mode == 3 && unified);
        if ((ret == 0) != success)
        {
            fprintf(stderr, "budget boundary mode %d returned %d\n", mode, ret);
            return -1;
        }
        const bool managed = ncnn::get_weight_cache_limits(&store, remaining_device, remaining_host, preferred_device, preferred_host) > 0;
        if (mode == 5 && (managed || weight_resident_size(weights, 2, 0) != 4096))
            return -1;
        if (mode == 1 && weight_resident_size(weights, 2, 0) + preferred_device > device_budget)
            return -1;
    }
    return 0;
}

static int test_transfer_submit_failure()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    for (int second = 0; second < 2; second++)
    {
        if (second && vkdev->info.unified_compute_transfer_queue())
            continue;
        ncnn::VkWeightAllocator weights(vkdev);
        ncnn::VkWeightStagingAllocator staging(vkdev);
        ncnn::Option opt;
        opt.use_fp16_packed = opt.use_fp16_storage = false;
        opt.blob_vkallocator = &weights;
        opt.staging_vkallocator = &staging;
        ncnn::Mat input(128);
        input.fill(1.f);
        ncnn::VkMat output;
        ncnn::VkTransfer cmd(vkdev);
        cmd.record_upload(input, output, opt);
        WeightAllocationAudit audit(vkdev, second ? WeightAllocationAudit::FAIL_SECOND_SUBMIT : WeightAllocationAudit::FAIL_SUBMIT);
        if (cmd.submit_and_wait() == 0 || !audit.failed || !cmd.error() || cmd.reset() == 0
                || audit.submissions != (uint32_t)second + 1 || audit.waited_submissions != (uint32_t)second || audit.queue_held_while_waiting)
            return -1;
    }
    return 0;
}

static int test_exchange_dependencies()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator store(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    if (store.set_weights_in_host_disk(true))
        return -1;
    ncnn::VkMat weights[5];
    ncnn::Mat expected[5];
    for (int i = 0; i < 5; i++)
    {
        expected[i] = RandomMat(256);
        weights[i].create(256, (size_t)4u, &store);
        if (weights[i].empty())
            return -1;
        memcpy(weights[i].data->external_ptr, expected[i].data, 1024);
    }
    if (store.finalize_weights())
        return -1;
    ncnn::Option opt;
    opt.use_fp16_packed = opt.use_fp16_storage = false;
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_device_cache_size = 2048;
    opt.weight_host_cache_size = 256;
    ncnn::VkCompute cmd(vkdev, opt);
    ncnn::Mat outputs[5];
    for (int i = 0; i < 5; i++)
        cmd.record_clone(weights[i], outputs[i], opt);
    reset_weight_benchmark();
    WeightAllocationAudit audit(vkdev);
    if (cmd.submit_and_wait() || audit.queue_held_while_waiting)
        return -1;
    for (int i = 0; i < 5; i++)
        if (CompareMat(expected[i], outputs[i], 0.f))
            return -1;
#if NCNN_BENCHMARK
    // Five readers, the ordinary prefix (if present), and the final handoff.
    // Returning overwritten regions must not add submissions behind newer work.
    const uint64_t expected_submits = 6 + (vkdev->info.support_VK_KHR_push_descriptor() ? 1 : 0);
    if (weight_benchmark_counts()["compute_submissions"] != expected_submits)
        return -1;
#endif
    return 0;
}

static int test_automatic_capacity_limit()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator store(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    if (store.set_weights_in_host_disk(true))
        return -1;
    ncnn::VkMat weight(12288, (size_t)4u, &store);
    ncnn::Mat input = RandomMat(12288);
    if (weight.empty())
        return -1;
    memcpy(weight.data->external_ptr, input.data, 49152);
    if (store.finalize_weights())
        return -1;
    ncnn::Option opt;
    opt.use_fp16_packed = opt.use_fp16_storage = false;
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_device_budget = opt.weight_host_budget = 65536;
    for (int fixed = 0; fixed < 2; fixed++)
    {
        opt.weight_device_cache_size = fixed ? 98304 : 0;
        ncnn::VkCompute cmd(vkdev, opt);
        ncnn::Mat output;
        cmd.record_clone(weight, output, opt);
        const int ret = cmd.submit_and_wait();
        if (fixed ? ret == 0 : ret != 0 || CompareMat(input, output, 0.f))
            return -1;
    }
    return 0;
}

static int test_allocation_generation()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    for (int mode = 0; mode < 3; mode++)
    {
        ncnn::VkWeightAllocator store(vkdev, mode == 1);
        if (mode == 2 && store.set_weights_in_host_disk(true))
            return -1;
        ncnn::VkMat old(128, (size_t)4u, &store);
        if (old.empty())
            return -1;
        store.clear();
        ncnn::VkMat current(128, (size_t)4u, &store);
        if (current.empty() || store.finalize_weights())
            return -1;
        size_t db, hb, ds, hs;
        const int managed = ncnn::get_weight_cache_limits(&store, db, hb, ds, hs);
        uint64_t before, after;
        if (ncnn::get_weight_source_generation(&store, before) < 0)
            return -1;
        old.release();
        if (ncnn::get_weight_source_generation(&store, after) < 0 || before != after
                || ncnn::get_weight_cache_limits(&store, db, hb, ds, hs) != managed)
            return -1;
        current.release();
        if (ncnn::get_weight_cache_limits(&store, db, hb, ds, hs))
            return -1;
    }
    return 0;
}

static int test_automatic_residency()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    for (int mode = 0; mode < 2; mode++)
    {
        ncnn::VkWeightAllocator store(vkdev);
        const size_t unit = 1024 * 1024;
        const size_t budget = mode == 0 ? 10 * unit : 6 * unit;
        if (store.set_weights_in_host_disk(true) || store.set_weight_budgets(budget, budget))
            return -1;
        ncnn::VkMat weights[3];
        for (int i = 0; i < 3; i++)
        {
            ncnn::begin_weight_upload(&store);
            const size_t size = mode == 0 ? (i < 2 ? 5 * unit : 0) : (i == 0 ? unit : 4 * unit);
            if (!size)
                continue;
            weights[i].create((int)(size / 4), (size_t)4u, &store);
            if (weights[i].empty())
                return -1;
            memset(weights[i].data->external_ptr, 0, size);
        }
        if (store.finalize_weights())
            return -1;
        size_t db, hb, ds, hs;
        const int managed = ncnn::get_weight_cache_limits(&store, db, hb, ds, hs);
        if (mode == 0 && (managed || weight_resident_size(weights, 3, 0) != budget))
            return -1;
        if (mode == 1 && (managed <= 0 || weights[0].data->storage_type != 0 || ds < 4 * unit || ds > db))
            return -1;
    }
    return 0;
}

static int test_weight_storage(int argc, char** argv)
{
    if (ncnn::get_gpu_count() == 0)
        return 0;
    gpu_index = ncnn::get_default_gpu_index();
    if (argc > 1)
        gpu_index = atoi(argv[1]);
    if (gpu_index < 0 || gpu_index >= ncnn::get_gpu_count())
        return -1;

    if (test_transfer_submit_failure() || test_exchange_dependencies() || test_automatic_capacity_limit() || test_allocation_generation() || test_automatic_residency() || test_external_sources()
            || test_concurrent_reads() || test_upload_error() || test_resident_layout(false) || test_resident_layout(true) || test_host_cache() || test_budget_boundaries() || test_native_allocation_budget())
        return -1;
#if defined(__linux__)
    if (test_file_creation_fallback() || test_file_directory() || test_file_write_failure())
        return -1;
#endif
    for (int mode = 0; mode < 3; mode++)
        for (int pack = 1; pack <= 4; pack *= 4)
            for (int flatten = 0; flatten < 2; flatten++)
                for (int host = 0; host < 2; host++)
                {
                    if (test_prepared_bytes(mode, pack, flatten != 0, 1, host != 0))
                        return -1;
#if NCNN_BATCH
                    if (test_prepared_bytes(mode, pack, flatten != 0, 3, host != 0))
                        return -1;
#endif
                }
    return 0;
}

int main(int argc, char** argv)
{
    const int ret = test_weight_storage(argc, argv);
    ncnn::destroy_gpu_instance();
    return ret;
}
