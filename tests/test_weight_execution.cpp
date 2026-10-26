// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "command.h"
#include "datareader.h"
#include "gpu.h"
#include "net.h"
#include "pipeline.h"
#include "layer.h"
#include "modelbin.h"
#include "testutil.h"

#if defined(NCNN_WEIGHT_TEST_INTERNAL)
namespace ncnn {
void begin_weight_upload(VkWeightAllocator* allocator);
}
#endif

static int gpu_index;
static int precision;

static ncnn::Option weight_option()
{
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_fp16_storage = false;
    opt.use_fp16_packed = precision == 1;
    opt.use_fp16_storage = precision == 1;
    opt.use_bf16_packed = precision == 2;
    opt.use_bf16_storage = precision == 2;
    opt.use_fp16_arithmetic = false;
    opt.use_cooperative_matrix = false;
    return opt;
}

struct ExtractTask
{
    const ncnn::Net* net;
    const ncnn::Mat* input;
    ncnn::Mat output;
    int ret;
};

static void* extract_thread(void* args)
{
    ExtractTask* task = (ExtractTask*)args;
    ncnn::Extractor ex = task->net->create_extractor();
    task->ret = ex.input("b0", *task->input) || ex.extract("b12", task->output);
    return 0;
}

static int run_net(int storage, const ncnn::Mat& input, ncnn::Mat& output, bool small_cache)
{
    ncnn::Net net;
    net.opt = weight_option();
    net.set_vulkan_device(gpu_index);
    net.opt.use_weights_in_host_memory = storage == 1;
    net.opt.use_weights_in_host_disk = storage >= 2;
    net.opt.weight_host_cache_size = small_cache ? 16 : 2048;
    net.opt.weight_device_cache_size = small_cache ? 16 : 2048;
    const size_t weight_unit = precision ? 512 : 1024;
    if (storage >= 3)
    {
        net.opt.weight_device_budget = storage == 3 ? 6 * weight_unit : 65536;
        net.opt.weight_host_budget = storage == 3 ? 6 * weight_unit : 65536;
        net.opt.weight_device_cache_size = 2 * weight_unit;
        net.opt.weight_host_cache_size = weight_unit;
    }

    std::string param = "7767517\n13 13\nInput input 0 1 b0\n";
    std::vector<unsigned char> model;
    for (int i = 0; i < 12; i++)
    {
        char line[256];
        snprintf(line, sizeof(line), "Gemm gemm%d 1 1 b%d b%d 5=1 6=1 8=16 9=16 10=-1\n", i, i, i + 1);
        param += line;
        const size_t base = model.size();
        model.resize(base + 4 + 16 * 16 * 4);
        const uint32_t tag = 0;
        memcpy(model.data() + base, &tag, 4);
        for (int j = 0; j < 16 * 16; j++)
        {
            float v = j / 16 == j % 16 ? 1.f + (i % 3 - 1) * 0.125f : 0.f;
            memcpy(model.data() + base + 4 + j * 4, &v, 4);
        }
    }
    if (net.load_param_mem(param.c_str()))
        return -1;
    const unsigned char* model_ptr = model.data();
    ncnn::DataReaderFromMemory reader(model_ptr);
#if defined(__linux__)
    if (storage == 3 && precision == 0)
    {
        const char* directory = getenv("TMPDIR");
        const bool had_directory = directory != 0;
        const std::string previous_directory = directory ? directory : "";
        if (setenv("TMPDIR", "/dev/null/ncnn-weights", 1))
            return -1;
        const int failed_load = net.load_model(reader);
        if (had_directory)
            setenv("TMPDIR", previous_directory.c_str(), 1);
        else
            unsetenv("TMPDIR");
        if (!failed_load)
            return -1;
    }
#endif
    if (net.load_model(reader))
        return -1;

    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("b0", input))
        return -1;
    int ret = ex.extract("b12", output);
    if (small_cache)
        return ret != 0 ? 0 : -1;
    if (ret)
        return ret;

    if (storage == 3 && precision == 0)
    {
        for (int copy = 0; copy < 3; copy++)
        {
            ncnn::Extractor oversized = copy == 1 ? ex : net.create_extractor();
            if (copy == 2)
                oversized = ex;
            // clear copied activations so the new command must execute its weights
            for (int i = 0; i < 13; i++)
                if (oversized.input(i, ncnn::Mat()) || oversized.input(i, ncnn::VkMat()))
                    return -1;
            oversized.set_weight_device_cache_size(65536);
            ncnn::Mat invalid;
            if (oversized.input("b0", input) || oversized.extract("b12", invalid) == 0)
                return -1;
        }
    }

    // another extractor shares the model and uses an independent command
    ncnn::Extractor ex2 = net.create_extractor();
    ex2.set_weight_host_cache_size(storage >= 3 ? weight_unit : 4096);
    ex2.set_weight_device_cache_size(storage >= 3 ? 2 * weight_unit : 4096);
    ncnn::Mat second;
    if (ex2.input("b0", input) || ex2.extract("b12", second) || CompareMat(output, second, 0.0001f))
        return -1;
    if (storage == 2 && precision == 0)
    {
        ncnn::Extractor cached = net.create_extractor();
        cached.set_weight_device_cache_size(65536);
        ncnn::Mat first;
        if (cached.input("b0", input) || cached.extract("b12", first) || CompareMat(output, first, 0.0001f))
            return -1;
        const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
        ncnn::VkBlobAllocator blobs(vkdev);
        ncnn::VkStagingAllocator staging(vkdev);
        ncnn::Option download_opt = weight_option();
        download_opt.blob_vkallocator = &blobs;
        download_opt.staging_vkallocator = &staging;
        download_opt.use_packing_layout = false;
        download_opt.weight_device_cache_size = 65536;
        ncnn::VkCompute cmd(vkdev, download_opt);
        for (int round = 0; round < 2; round++)
        {
            for (int i = 0; i < 13; i++)
                if (cached.input(i, ncnn::Mat()) || cached.input(i, ncnn::VkMat()))
                    return -1;
            if (round && cmd.reset())
                return -1;
            ncnn::VkMat gpu_output;
            ncnn::Mat external;
            if (cached.input("b0", input) || cached.extract("b12", gpu_output, cmd))
                return -1;
            cmd.record_download(gpu_output, external, download_opt);
            if (cmd.submit_and_wait() || CompareMat(output, external, 0.0001f))
                return -1;
#if NCNN_BENCHMARK && defined(NCNN_WEIGHT_TEST_INTERNAL)
            std::map<std::string, uint64_t>& stats = weight_benchmark_counts();
            if (stats["file_reads"] != 12 || stats["cache_hits"] < (uint64_t)round * 12)
                return -1;
#endif
        }
    }
#if NCNN_THREADS
    if (storage != 0)
    {
        ExtractTask tasks[2];
        for (int i = 0; i < 2; i++)
        {
            tasks[i].net = &net;
            tasks[i].input = &input;
            tasks[i].ret = -1;
        }
        ncnn::Thread thread0(extract_thread, &tasks[0]);
        ncnn::Thread thread1(extract_thread, &tasks[1]);
        thread0.join();
        thread1.join();
        for (int i = 0; i < 2; i++)
            if (tasks[i].ret || CompareMat(output, tasks[i].output, 0.0001f))
                return -1;
    }
#endif
    return 0;
}

static int test_network()
{
    ncnn::Mat input = RandomMat(16, 8);
    ncnn::Mat device;
    ncnn::Mat host;
    ncnn::Mat disk;
    if (run_net(0, input, device, false) || run_net(1, input, host, false) || run_net(2, input, disk, false))
        return -1;
    if (CompareMat(device, host, 0.0001f) || CompareMat(device, disk, 0.0001f))
        return -1;
    for (int storage = 3; storage < 5; storage++)
    {
        ncnn::Mat resident;
        if (run_net(storage, input, resident, false) || CompareMat(device, resident, 0.0001f))
            return -1;
    }
    ncnn::Mat expected = input.clone();
    for (size_t i = 0; i < expected.total(); i++)
    {
        float& v = ((float*)expected)[i];
        for (int j = 0; j < 12; j++)
            v *= 1.f + (j % 3 - 1) * 0.125f;
    }
    if (CompareMat(expected, disk, precision == 2 ? 0.03f : precision == 1 ? 0.003f : 0.0001f))
        return -1;
    return run_net(2, input, disk, true);
}

static int test_empty_upload()
{
    const char* param = "7767517\n2 2\nInput input 0 1 input\nPriorBox prior 1 1 input output -23300=1,1.0 -23301=0 -23302=0 9=64 10=64\n";
    ncnn::Mat input = RandomMat(4, 4, 1);
    ncnn::Mat reference;
    for (int vulkan = 0; vulkan < 2; vulkan++)
    {
        ncnn::Net net;
        net.opt = weight_option();
        net.opt.use_vulkan_compute = vulkan != 0;
        net.set_vulkan_device(gpu_index);
        const unsigned char bytes[4] = {};
        const unsigned char* ptr = bytes;
        ncnn::DataReaderFromMemory reader(ptr, sizeof(bytes));
        if (net.load_param_mem(param) || net.load_model(reader))
            return -1;
        ncnn::Extractor ex = net.create_extractor();
        ncnn::Mat output;
        if (ex.input("input", input) || ex.extract("output", output))
            return -1;
        if (!vulkan)
            reference = output;
        else if (CompareMat(reference, output, 0.0001f))
            return -1;
    }
    return 0;
}

static int test_standalone_sources()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkWeightStagingAllocator upload_staging(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    if (weights.set_weights_in_host_disk(true))
        return -1;
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &upload_staging;
    ncnn::Mat original = RandomMat(32);
    ncnn::VkMat file;
    {
        ncnn::VkTransfer upload(vkdev);
        upload.record_upload(original, file, opt);
        if (upload.submit_and_wait() || weights.finalize_weights())
            return -1;
    }
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_device_cache_size = 1024;
    opt.weight_host_cache_size = 1024;
    ncnn::VkCompute cmd(vkdev);
    ncnn::Mat output;
    cmd.record_clone(file, output, opt);
    if (cmd.submit_and_wait() || CompareMat(original, output, 0.0001f) || cmd.reset())
        return -1;

    // Packing can alias a FILE source; downloading it still requires real staging.
    opt.use_packing_layout = false;
    cmd.record_download(file, output, opt);
    if (cmd.submit_and_wait() || CompareMat(original, output, 0.0001f) || cmd.reset())
        return -1;

    // A borrowed HOST source has no Vulkan buffer and needs no weight allocator.
    ncnn::VkBufferMemory source;
    source.buffer = 0;
    source.memory = 0;
    source.offset = 0;
    source.capacity = original.total() * original.elemsize;
    source.external_ptr = original.data;
    source.external_size = source.capacity;
    source.memory_type_index = (uint32_t)-1;
    source.access_flags = VK_ACCESS_HOST_WRITE_BIT;
    source.stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    source.refcount = 0;
    source.storage_type = 1;
    ncnn::VkMat host(32, &source, 4u, &blobs);
    for (int round = 0; round < 2; round++)
    {
        // Borrowed bytes have no model generation; invalidate before changing them.
        if (cmd.clear_weight_cache())
            return -1;
        original.fill((float)(round + 1));
        cmd.record_clone(host, output, opt);
        if (cmd.submit_and_wait() || CompareMat(original, output, 0.0001f) || cmd.reset())
            return -1;
    }
    return 0;
}

#if defined(NCNN_WEIGHT_TEST_INTERNAL)
static int test_copy_residency()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator store(vkdev);
    ncnn::VkWeightStagingAllocator upload_staging(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    if (store.set_weights_in_host_disk(true))
        return -1;
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &store;
    opt.staging_vkallocator = &upload_staging;
    ncnn::VkMat weights[3];
    ncnn::Mat original[3];
    {
        ncnn::VkTransfer upload(vkdev);
        for (int i = 0; i < 3; i++)
        {
            original[i] = RandomMat(256);
            upload.record_upload(original[i], weights[i], opt);
        }
        if (upload.submit_and_wait() || store.finalize_weights())
            return -1;
    }
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_host_cache_size = 1024;
    opt.weight_device_cache_size = 2048;
    reset_weight_benchmark();
    ncnn::VkCompute cmd(store, opt);
    const int order[6] = {0, 1, 0, 2, 0, 1};
    ncnn::Mat outputs[6];
    for (int i = 0; i < 6; i++)
        cmd.record_clone(weights[order[i]], outputs[i], opt);
#if NCNN_BENCHMARK
    std::map<std::string, uint64_t>& stats = weight_benchmark_counts();
    if (stats["bytes_read"] || stats["bytes_uploaded"] || weight_benchmark_counts()["host_capacity"])
        return -1;
#endif
    if (cmd.submit_and_wait())
        return -1;
    for (int i = 0; i < 6; i++)
        if (CompareMat(original[order[i]], outputs[i], 0.f))
            return -1;
#if NCNN_BENCHMARK
    if (stats["file_reads"] < 3 || weight_benchmark_counts()["device_capacity"] != 2048)
        return -1;
    if (weight_benchmark_counts()["unified_memory"])
    {
        if (weight_benchmark_counts()["host_capacity"] || stats["bytes_uploaded"] || stats["unified_bytes_written"] != stats["bytes_read"])
            return -1;
    }
    else if (stats["bytes_uploaded"] < stats["bytes_read"] || weight_benchmark_counts()["host_capacity"] != 1024)
        return -1;
#endif

    if (cmd.reset())
        return -1;
    ncnn::Mat repeated;
    cmd.record_clone(weights[1], repeated, opt);
#if NCNN_BENCHMARK
    const uint64_t reads = stats["file_reads"];
#endif
    if (cmd.submit_and_wait() || CompareMat(original[1], repeated, 0.f))
        return -1;
#if NCNN_BENCHMARK
    if (stats["file_reads"] != reads || !stats["cache_hits"])
        return -1;
#endif

#if NCNN_BATCH
    // a view retains the allocation identity but resolves its own offset
    if (cmd.reset())
        return -1;
    ncnn::Mat view_result;
    ncnn::VkMat view(128, weights[0].data, (size_t)4u, &store);
    view.offset = 64 * sizeof(float);
    cmd.record_download(view, view_result, opt);
    if (cmd.submit_and_wait() || CompareMat(original[0].range(64, 128), view_result, 0.f))
        return -1;

#endif

    if (vkdev->info.support_image_storage())
    {
        if (cmd.reset())
            return -1;
        ncnn::VkImageMat image;
        ncnn::Mat image_result;
        cmd.record_clone(weights[2], image, opt);
        cmd.record_clone(image, image_result, opt);
        image.release();
        if (cmd.submit_and_wait() || CompareMat(original[2], image_result, 0.f))
            return -1;
    }
    return 0;
}

class CountHostAllocator : public ncnn::VkStagingAllocator
{
public:
    explicit CountHostAllocator(const ncnn::VulkanDevice* vkdev)
        : ncnn::VkStagingAllocator(vkdev), allocations(0), frees(0), flushes(0), peak(0)
    {
        VkBuffer probe = create_buffer(4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        VkMemoryRequirements requirements;
        ncnn::vkGetBufferMemoryRequirements(vkdev->vkdevice(), probe, &requirements);
        buffer_memory_type_index = vkdev->find_memory_index(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        ncnn::vkDestroyBuffer(vkdev->vkdevice(), probe, 0);
        // exercise explicit flushes even on devices whose host memory is coherent
        coherent = false;
    }

    virtual ncnn::VkBufferMemory* fastMalloc(size_t size)
    {
        allocations++;
        ncnn::VkBufferMemory* p = ncnn::VkStagingAllocator::fastMalloc(size);
        if (p)
        {
            VkMemoryRequirements requirements;
            ncnn::vkGetBufferMemoryRequirements(vkdev->vkdevice(), p->buffer, &requirements);
            peak = std::max(peak, (size_t)requirements.size);
        }
        return p;
    }

    virtual void fastFree(ncnn::VkBufferMemory* p)
    {
        frees++;
        ncnn::VkStagingAllocator::fastFree(p);
    }

    virtual int flush(ncnn::VkBufferMemory* p)
    {
        flushes++;
        if (p->offset % vkdev->info.non_coherent_atom_size() || p->capacity % vkdev->info.non_coherent_atom_size())
            return -1;
        return ncnn::VkStagingAllocator::flush(p);
    }

    int allocations;
    int frees;
    int flushes;
    size_t peak;
};

class CountDeviceAllocator : public ncnn::VkBlobAllocator
{
public:
    explicit CountDeviceAllocator(const ncnn::VulkanDevice* vkdev, bool unified = false)
        : ncnn::VkBlobAllocator(vkdev, 4096), allocations(0), frees(0), flushes(0), peak(0)
    {
        if (unified)
        {
            VkBuffer probe = create_buffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            VkMemoryRequirements requirements;
            ncnn::vkGetBufferMemoryRequirements(vkdev->vkdevice(), probe, &requirements);
            buffer_memory_type_index = vkdev->find_memory_index(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0, 0);
            ncnn::vkDestroyBuffer(vkdev->vkdevice(), probe, 0);
            mappable = true;
            coherent = false;
        }
    }

    virtual ncnn::VkBufferMemory* fastMalloc(size_t size)
    {
        allocations++;
        ncnn::VkBufferMemory* p = ncnn::VkBlobAllocator::fastMalloc(size);
        if (p)
        {
            VkMemoryRequirements requirements;
            ncnn::vkGetBufferMemoryRequirements(vkdev->vkdevice(), p->buffer, &requirements);
            peak = std::max(peak, (size_t)requirements.size);
        }
        return p;
    }

    virtual void fastFree(ncnn::VkBufferMemory* p)
    {
        frees++;
        ncnn::VkBlobAllocator::fastFree(p);
    }

    virtual int flush(ncnn::VkBufferMemory* p)
    {
        flushes++;
        return ncnn::VkBlobAllocator::flush(p);
    }

    int allocations;
    int frees;
    int flushes;
    size_t peak;
};

static int test_cache_reuse(int queue_mode)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    CountHostAllocator host(vkdev);
    CountDeviceAllocator device(vkdev, unified);
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_host_vkallocator = &host;
    opt.weight_device_vkallocator = &device;
    opt.weight_host_cache_size = 2048;
    opt.weight_device_cache_size = 2048;
    opt.use_weights_in_host_disk = true;
    opt.use_packing_layout = false;
    ncnn::Net net;
    net.opt = opt;
    net.set_vulkan_device(gpu_index);
    const char param[] = "7767517\n3 3\n"
                         "MemoryData weight0 0 1 w0 0=256\n"
                         "MemoryData weight1 0 1 w1 0=256\n"
                         "MemoryData weight2 0 1 w2 0=256\n";
    ncnn::Mat original[3];
    std::vector<unsigned char> model(3 * 256 * sizeof(float));
    for (int i = 0; i < 3; i++)
    {
        original[i] = RandomMat(256);
        memcpy(model.data() + i * 256 * sizeof(float), original[i].data, 256 * sizeof(float));
    }
    const unsigned char* model_ptr = model.data();
    ncnn::DataReaderFromMemory reader(model_ptr);
    if (net.load_param_mem(param) || net.load_model(reader))
        return -1;
    {
        ncnn::VkCompute survivor(vkdev, opt, queue_mode);
        reset_weight_benchmark();
        const int order[] = {0, 1, 0, 2, 1, 0};
        for (int i = 0; i < 6; i++)
        {
            if (i && survivor.reset())
                return -1;
            // Newly created Extractors reuse the supplied command's working set.
            ncnn::Extractor ex = net.create_extractor();
#if NCNN_BENCHMARK
            std::map<std::string, uint64_t>& stats = weight_benchmark_counts();
            const uint64_t reads = stats["file_reads"];
            const uint64_t hits = stats["cache_hits"];
#endif
            ncnn::VkMat gpu_output;
            ncnn::Mat output;
            if (ex.extract(order[i], gpu_output, survivor))
                return -1;
            survivor.record_download(gpu_output, output, opt);
            if (survivor.submit_and_wait() || CompareMat(original[order[i]], output, 0.f))
                return -1;
            if (device.allocations != 1 || device.frees || host.allocations != (unified ? 0 : 1) || host.frees)
                return -1;
#if NCNN_BENCHMARK
            if ((i == 2 || i == 4) && (stats["file_reads"] != reads || stats["cache_hits"] <= hits))
                return -1;
#endif
        }
        // Extractor copy/assignment shares sources, never another command's arenas.
        ncnn::Extractor original_extractor = net.create_extractor();
        for (int copy = 0; copy < 2; copy++)
        {
            ncnn::Extractor other = copy == 0 ? original_extractor : net.create_extractor();
            if (copy == 1)
                other = original_extractor;
            ncnn::VkCompute isolated(vkdev, opt, queue_mode);
            ncnn::VkMat gpu_output;
            ncnn::Mat output;
            if (other.extract(0, gpu_output, isolated))
                return -1;
            isolated.record_download(gpu_output, output, opt);
            if (isolated.submit_and_wait() || CompareMat(original[0], output, 0.f)
                    || device.allocations != copy + 2 || device.frees != copy)
                return -1;
#if NCNN_BENCHMARK
            if (weight_benchmark_counts()["file_reads"] != 1)
                return -1;
#endif
        }
        {
            // Mat extraction has its own command, independently of external ones.
            ncnn::Extractor internal = net.create_extractor();
            for (int round = 0; round < 2; round++)
            {
                ncnn::Mat output;
                if (internal.input(0, ncnn::Mat()) || internal.input(0, ncnn::VkMat())
                        || internal.extract(0, output) || CompareMat(original[0], output, 0.f))
                    return -1;
                if (device.allocations != 4 || device.frees != 2)
                    return -1;
#if NCNN_BENCHMARK
                if (weight_benchmark_counts()["file_reads"] != 1)
                    return -1;
#endif
            }
        }
        // Reset retains arenas; explicit clearing returns them to their allocators.
        if (device.frees != 3 || host.frees != (unified ? 0 : 3) || survivor.reset())
            return -1;
        if (device.frees != 3 || survivor.clear_weight_cache())
            return -1;
        if (device.frees != 4 || host.frees != (unified ? 0 : 4))
            return -1;
        ncnn::Extractor ex = net.create_extractor();
        ncnn::VkMat gpu_output;
        ncnn::Mat output;
        if (ex.extract(0, gpu_output, survivor))
            return -1;
        survivor.record_download(gpu_output, output, opt);
        if (survivor.clear_weight_cache() == 0)
            return -1;
        if (survivor.submit_and_wait() || CompareMat(original[0], output, 0.f)
                || device.allocations != 5 || device.frees != 4
                || survivor.reset() || survivor.clear_weight_cache())
            return -1;
        if (device.frees != 5 || host.frees != (unified ? 0 : 5))
            return -1;
    }
    fprintf(stderr, "command cache reuse, eviction, isolation and explicit clearing verified mode=%d\n", queue_mode);
    return 0;
}

static int test_queue_ring(int queue_mode)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
    ncnn::VkWeightAllocator store(vkdev);
    ncnn::VkWeightStagingAllocator upload_staging(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    CountHostAllocator host(vkdev);
    CountDeviceAllocator device(vkdev, unified);
    if (store.set_weights_in_host_disk(true))
        return -1;
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &store;
    opt.staging_vkallocator = &upload_staging;
    ncnn::VkMat weights[5];
    ncnn::Mat original[5];
    const int sizes[5] = {128, 192, 256, 384, 512};
    {
        ncnn::VkTransfer upload(vkdev);
        for (int i = 0; i < 5; i++)
        {
            original[i] = RandomMat(sizes[i]);
            upload.record_upload(original[i], weights[i], opt);
        }
        if (upload.submit_and_wait() || store.finalize_weights())
            return -1;
    }
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_host_vkallocator = &host;
    opt.weight_device_vkallocator = &device;
    opt.weight_host_cache_size = 2048;
    opt.weight_device_cache_size = 4096;
    {
        reset_weight_benchmark();
        ncnn::VkCompute cmd(store, opt, queue_mode);
        for (int round = 0; round < 3; round++)
        {
            if (round && cmd.reset())
                return -1;
            ncnn::Mat outputs[160];
            for (int i = 0; i < 160; i++)
                cmd.record_clone(weights[(i * 3 + round) % 5], outputs[i], opt);
            if (!round && (host.allocations || device.allocations))
                return -1;
            if (cmd.submit_and_wait())
                return -1;
            for (int i = 0; i < 160; i++)
                if (CompareMat(original[(i * 3 + round) % 5], outputs[i], 0.f))
                    return -1;
            if (host.allocations != (unified ? 0 : 1) || device.allocations != 1 || host.frees || device.frees)
                return -1;
        }
#if NCNN_BENCHMARK
        std::map<std::string, uint64_t>& stats = weight_benchmark_counts();
        if (stats["compute_submissions"] < 4)
            return -1;
        if (weight_benchmark_counts()["unified_memory"])
        {
            if (stats["transfer_submissions"] || stats["bytes_uploaded"] || !device.flushes || stats["unified_bytes_written"] != stats["bytes_read"])
                return -1;
        }
        else if (stats["transfer_submissions"] < 32 || !host.flushes)
            return -1;
        if (queue_mode && (weight_benchmark_counts()["separate_transfer_family"] || stats["ownership_transfers"]))
            return -1;
        if (queue_mode == 2 && weight_benchmark_counts()["separate_transfer_queue"])
            return -1;
        if (!weight_benchmark_counts()["unified_memory"] && queue_mode == 1 && vkdev->info.compute_queue_count() > 1 && !weight_benchmark_counts()["separate_transfer_queue"])
            return -1;
        if (!weight_benchmark_counts()["unified_memory"] && !queue_mode && vkdev->info.transfer_queue_family_index() != vkdev->info.compute_queue_family_index() && !stats["ownership_transfers"])
            return -1;
        fprintf(stderr, "weight queue mode=%d separate=%d family=%d transfers=%llu computes=%llu ownership=%llu pending=%u/%u host=%lu device=%lu\n",
                queue_mode, (int)weight_benchmark_counts()["separate_transfer_queue"], (int)weight_benchmark_counts()["separate_transfer_family"],
                (unsigned long long)stats["transfer_submissions"], (unsigned long long)stats["compute_submissions"], (unsigned long long)stats["ownership_transfers"],
                (unsigned int)stats["peak_pending_transfer"], (unsigned int)stats["peak_pending_compute"], (unsigned long)host.peak, (unsigned long)device.peak);
#endif
    }
    const size_t alignment = std::max(vkdev->info.memory_map_alignment(), vkdev->info.non_coherent_atom_size());
    if (host.frees != (unified ? 0 : 1) || device.frees != 1 || host.peak > 2048 + alignment || device.peak > 4096 + alignment)
        return -1;
    return 0;
}

static int test_host_queue_modes(bool resident)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator store(vkdev, !resident);
    if (vkdev->info.type() == 1 || vkdev->info.type() == 3)
        return 0;
    if (resident && (store.set_weights_in_host_disk(true) || store.set_weight_budgets(24576, 10240)))
        return -1;
    ncnn::VkWeightStagingAllocator upload_staging(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &store;
    opt.staging_vkallocator = &upload_staging;
    ncnn::VkMat weights[5];
    ncnn::Mat original[5];
    {
        ncnn::VkTransfer upload(vkdev);
        for (int i = 0; i < 5; i++)
        {
            ncnn::begin_weight_upload(&store);
            original[i] = RandomMat(256 << i);
            upload.record_upload(original[i], weights[i], opt);
        }
        if (upload.submit_and_wait() || store.finalize_weights(16384, 2048))
            return -1;
    }
    if (weights[3].data->storage_type != 1
            || (resident && weights[4].data->storage_type != 2))
        return -1;
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_device_cache_size = 16384;
    opt.weight_host_cache_size = 2048;
    // reuse the same immutable host source across transfer and compute families
    const int queue_modes[4] = {0, 1, 2, 0};
    for (int mode = 0; mode < 4; mode++)
    {
        reset_weight_benchmark();
        ncnn::VkCompute cmd(store, opt, queue_modes[mode]);
        for (int round = 0; round < 2; round++)
        {
            if (round && cmd.reset())
                return -1;
            ncnn::Mat outputs[20];
            for (int i = 0; i < 20; i++)
                cmd.record_clone(weights[(i + round) % 5], outputs[i], opt);
            if (cmd.submit_and_wait())
                return -1;
            for (int i = 0; i < 20; i++)
                if (CompareMat(original[(i + round) % 5], outputs[i], 0.f))
                    return -1;
        }
#if NCNN_BENCHMARK
        std::map<std::string, uint64_t>& stats = weight_benchmark_counts();
        const bool separate_family = queue_modes[mode] == 0 && vkdev->info.compute_queue_family_index() != vkdev->info.transfer_queue_family_index();
        if (weight_benchmark_counts()["separate_transfer_family"] != separate_family || !stats["host_resident_bytes_uploaded"]
                || (separate_family && !stats["ownership_transfers"]))
            return -1;
#endif
    }
    fprintf(stderr, "host queue modes verified resident=%d\n", resident);
    return 0;
}

class InvalidDeviceAllocator : public ncnn::VkBlobAllocator
{
public:
    InvalidDeviceAllocator(const ncnn::VulkanDevice* vkdev, int _failure)
        : ncnn::VkBlobAllocator(vkdev, 4096), failure(_failure)
    {
    }

    virtual ncnn::VkBufferMemory* fastMalloc(size_t size)
    {
        if (failure == 0)
            return 0;
        ncnn::VkBufferMemory* p = ncnn::VkBlobAllocator::fastMalloc(failure == 3 ? size + 8192 : size);
        if (p)
        {
            original_offset = p->offset;
            original_memory_type = p->memory_type_index;
        }
        if (p && failure == 1)
            p->buffer_usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (p && failure == 2)
            p->storage_type = 1;
        if (p && failure == 4)
            p->offset++;
        if (p && failure == 5)
            p->memory_type_index = (uint32_t)-1;
        if (p && failure == 6)
        {
            VkMemoryRequirements requirements;
            ncnn::vkGetBufferMemoryRequirements(vkdev->vkdevice(), p->buffer, &requirements);
            p->offset = (size_t)requirements.size;
        }
        return p;
    }

    virtual void fastFree(ncnn::VkBufferMemory* p)
    {
        p->offset = original_offset;
        p->memory_type_index = original_memory_type;
        ncnn::VkBlobAllocator::fastFree(p);
    }

    int failure;
    size_t original_offset;
    uint32_t original_memory_type;
};

static int test_execution_failures()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator store(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    if (store.set_weights_in_host_disk(true))
        return -1;
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_host_cache_size = 2048;
    opt.weight_device_cache_size = 4096;
    ncnn::VkMat written(256, (size_t)4u, &store);
    ncnn::VkMat unwritten(256, (size_t)4u, &store);
    ncnn::VkMat unused(2048, (size_t)4u, &store);
    ncnn::Mat source = RandomMat(256);
    if (written.empty() || !written.data->external_ptr || written.data->external_size < 1024
            || !(written.data->access_flags & VK_ACCESS_HOST_WRITE_BIT))
        return -1;
    memcpy(written.data->external_ptr, source.data, 1024);
    if (store.finalize_weights())
        return -1;
    {
        // ordinary device weights do not become dependencies of another store
        ncnn::VkWeightAllocator device_store(vkdev);
        ncnn::VkMat resident(16, (size_t)4u, &device_store);
        if (resident.empty())
            return -1;
        reset_weight_benchmark();
        ncnn::VkCompute cmd(store, opt);
        ncnn::Mat output;
        cmd.record_clone(resident, output, opt);
        if (cmd.submit_and_wait())
            return -1;
    }
    for (int failure = 0; failure < 7; failure++)
    {
        InvalidDeviceAllocator device(vkdev, failure);
        opt.weight_device_vkallocator = &device;
        reset_weight_benchmark();
        ncnn::VkCompute cmd(store, opt);
        ncnn::Mat result;
        cmd.record_clone(written, result, opt);
        if (cmd.submit_and_wait() == 0)
            return -1;
    }
    {
        InvalidDeviceAllocator device(vkdev, 0);
        opt.weight_device_vkallocator = &device;
        ncnn::VkCompute cmd(store, opt);
        ncnn::Mat output;
        cmd.record_clone(written, output, opt);
        if (cmd.submit_and_wait() == 0 || cmd.reset())
            return -1;
        device.failure = -1;
        cmd.record_clone(written, output, opt);
        if (cmd.submit_and_wait() || CompareMat(source, output, 0.0001f) || cmd.reset() || cmd.clear_weight_cache())
            return -1;
        cmd.record_clone(written, output, opt);
        if (cmd.submit_and_wait() || CompareMat(source, output, 0.0001f))
            return -1;
    }
    opt.weight_device_vkallocator = 0;
    for (int failure = WeightAllocationAudit::FAIL_ALLOCATE; failure <= WeightAllocationAudit::FAIL_SUBMIT; failure++)
    {
        ncnn::VkCompute cmd(store, opt);
        ncnn::Mat output;
        cmd.record_clone(written, output, opt);
        {
            WeightAllocationAudit audit(vkdev, (WeightAllocationAudit::Failure)failure);
            if (cmd.submit_and_wait() == 0 || !audit.failed || cmd.reset() || audit.current[0] || audit.current[1])
                return -1;
        }
        cmd.record_clone(written, output, opt);
        if (cmd.submit_and_wait() || CompareMat(source, output, 0.0001f) || cmd.reset() || cmd.clear_weight_cache())
            return -1;
    }
    {
        // a larger unvisited weight does not enlarge the actual submission dependency
        ncnn::Option small = opt;
        small.weight_device_cache_size = 1024;
        small.weight_host_cache_size = 256;
        reset_weight_benchmark();
        ncnn::VkCompute cmd(store, small);
        ncnn::Mat output;
        cmd.record_clone(written, output, small);
        if (cmd.submit_and_wait() || CompareMat(source, output, 0.0001f) || cmd.reset())
            return -1;
        cmd.record_clone(unused, output, small);
        if (cmd.submit_and_wait() == 0)
            return -1;
    }
    {
        // reject a missing external source before submitting any upload
        reset_weight_benchmark();
        ncnn::VkCompute cmd(store, opt);
        ncnn::Mat first;
        ncnn::Mat second;
        void* external_ptr = unwritten.data->external_ptr;
        unwritten.data->external_ptr = 0;
        cmd.record_clone(written, first, opt);
        cmd.record_clone(unwritten, second, opt);
        const int ret = cmd.submit_and_wait();
        unwritten.data->external_ptr = external_ptr;
        if (!ret || cmd.reset())
            return -1;
        cmd.record_clone(written, first, opt);
        if (cmd.submit_and_wait() || CompareMat(source, first, 0.0001f))
            return -1;
    }
    return 0;
}

#endif // NCNN_WEIGHT_TEST_INTERNAL

static int test_existing_model()
{
    const char* directory = getenv("NCNN_TEST_MODEL_DIR");
    const std::string model_dir = directory ? directory : "../../examples";
    const std::string model_file = model_dir + "/squeezenet_v1.1.bin";
    FILE* file = fopen(model_file.c_str(), "rb");
    if (!file)
    {
        fprintf(stderr, "squeezenet weight storage test skipped: examples model is unavailable\n");
        return 0;
    }
    fclose(file);
    ncnn::Mat input = RandomMat(227, 227, 3);
    ncnn::Mat reference;
    ncnn::Net net;
    for (int storage = 0; storage < 4; storage++)
    {
        net.clear();
        net.opt = weight_option();
        net.set_vulkan_device(gpu_index);
        net.opt.use_weights_in_host_memory = storage == 1;
        net.opt.use_weights_in_host_disk = storage >= 2;
        net.opt.weight_host_cache_size = 4 * 1024 * 1024;
        net.opt.weight_device_cache_size = 8 * 1024 * 1024;
        if (storage == 3)
        {
            net.opt.weight_device_budget = 5 * 1024 * 1024;
            net.opt.weight_host_budget = 4 * 1024 * 1024;
            net.opt.weight_device_cache_size = 3 * 1024 * 1024;
            net.opt.weight_host_cache_size = 128 * 1024;
        }
        if (net.load_param((model_dir + "/squeezenet_v1.1.param").c_str()) || net.load_model(model_file.c_str()))
            return -1;
        ncnn::Extractor ex = net.create_extractor();
        ncnn::Mat output;
        if (ex.input("data", input) || ex.extract("prob", output))
            return -1;
        if (storage == 0)
            reference = output;
        else if (CompareMat(reference, output, 0.001f))
            return -1;
    }
    fprintf(stderr, "squeezenet device/host/file/resident outputs match\n");
    return 0;
}

class MixedWeightAllocator : public ncnn::VkWeightAllocator
{
public:
    MixedWeightAllocator(const ncnn::VulkanDevice* vkdev, bool host)
        : ncnn::VkWeightAllocator(vkdev, host), allocations(0)
    {
    }

    virtual ncnn::VkBufferMemory* fastMalloc(size_t size)
    {
        if (set_weights_in_host_disk(allocations++ % 2 == 0))
            return 0;
        ncnn::VkBufferMemory* memory = ncnn::VkWeightAllocator::fastMalloc(size);
        buffers.push_back(memory);
        return memory;
    }

    std::vector<ncnn::VkBufferMemory*> buffers;
    int allocations;
};

struct WeightTestLayer
{
    WeightTestLayer()
        : layer(0)
    {
    }
    ~WeightTestLayer()
    {
        if (layer)
        {
            layer->destroy_pipeline(opt);
            delete layer;
        }
    }
    ncnn::Layer* layer;
    ncnn::Option opt;
};

static int test_mixed_bindings(bool host, bool small_cache = false)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    MixedWeightAllocator weights(vkdev, host);
    ncnn::VkWeightStagingAllocator upload_staging(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &blobs;
    opt.workspace_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_host_cache_size = 2048;
    opt.weight_device_cache_size = small_cache ? 1024 : 2048;
    WeightTestLayer holder;
    holder.opt = opt;
    holder.layer = ncnn::create_layer_vulkan("Gemm");
    ncnn::Layer* layer = holder.layer;
    if (!layer)
        return -1;
    layer->vkdev = vkdev;
    ncnn::ParamDict pd;
    pd.set(4, 1);
    pd.set(5, 1);
    pd.set(6, 1);
    pd.set(7, 8);
    pd.set(8, 16);
    pd.set(9, 16);
    pd.set(10, -1);
    ncnn::Mat sources[2];
    sources[0] = RandomMat(16, 8);
    sources[1].create(16, 16);
    sources[1].fill(0.f);
    for (int i = 0; i < 16; i++)
        sources[1].row(i)[i] = 1.f;
    ncnn::ModelBinFromMatArray reader(sources);
    if (layer->load_param(pd) || layer->load_model(reader) || layer->create_pipeline(opt))
        return -1;
    {
        ncnn::VkTransfer upload(vkdev);
        ncnn::Option upload_opt = opt;
        upload_opt.blob_vkallocator = &weights;
        upload_opt.workspace_vkallocator = &weights;
        upload_opt.staging_vkallocator = &upload_staging;
        if (layer->upload_model(upload, upload_opt) || upload.submit_and_wait() || weights.finalize_weights())
            return -1;
    }
    if (weights.buffers.size() != 2 || weights.buffers[0]->storage_type != 2 || weights.buffers[1]->storage_type == 2)
        return -1;
    ncnn::VkCompute cmd(weights, opt);
    std::vector<ncnn::VkMat> bottom;
    std::vector<ncnn::VkMat> top(1);
    if (layer->forward(bottom, top, cmd, opt))
        return -1;
    ncnn::Mat output;
    opt.use_packing_layout = false;
    cmd.record_download(top[0], output, opt);
    const int ret = cmd.submit_and_wait();
    if (small_cache && weights.buffers[1]->external_size
            && weights.buffers[1]->storage_type != 0)
    {
        return ret ? 0 : -1;
    }
    if (ret || CompareMat(sources[0], output, 0.0001f))
        return -1;
    return 0;
}

static int test_automatic_weight_cache()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkWeightStagingAllocator upload_staging(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    if (weights.set_weights_in_host_disk(true))
        return -1;
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    ncnn::Mat sources[3];
    ncnn::VkMat prepared[3];
    {
        ncnn::VkTransfer upload(vkdev);
        ncnn::Option upload_opt = opt;
        upload_opt.blob_vkallocator = &weights;
        upload_opt.staging_vkallocator = &upload_staging;
        for (int i = 0; i < 3; i++)
        {
            sources[i] = RandomMat(256);
            upload.record_upload(sources[i], prepared[i], upload_opt);
        }
        if (upload.submit_and_wait())
            return -1;
    }
    {
        ncnn::VkCompute early(weights, opt);
        ncnn::Mat output;
        early.record_clone(prepared[0], output, opt);
        if (early.submit_and_wait() == 0 || early.reset())
            return -1;
        early.record_clone(prepared[0], output, opt);
        if (early.submit_and_wait() == 0)
            return -1;
    }
    // Manual uploads need no layer grouping or cache-options query.
    if (weights.finalize_weights())
        return -1;
    const char shader[] = "#version 450\n"
                          "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
                          "layout(binding = 0) readonly buffer A { float a[]; };\n"
                          "layout(binding = 1) readonly buffer B { float b[]; };\n"
                          "layout(binding = 2) readonly buffer C { float c[]; };\n"
                          "layout(binding = 3) writeonly buffer D { float d[]; };\n"
                          "void main() { uint i = gl_GlobalInvocationID.x; if (i < 256) d[i] = a[i] + b[i] + c[i]; }\n";
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(shader, opt, spirv))
        return -1;
    ncnn::Pipeline pipeline(vkdev);
    pipeline.set_local_size_xyz(1, 1, 1);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>()))
        return -1;
#if NCNN_BENCHMARK && defined(NCNN_WEIGHT_TEST_INTERNAL)
    reset_weight_benchmark();
#endif
    ncnn::VkCompute cmd(weights, opt);
    ncnn::Mat warm_output;
    cmd.record_clone(prepared[0], warm_output, opt);
    if (cmd.submit_and_wait() || CompareMat(sources[0], warm_output, 0.f) || cmd.reset())
        return -1;
#if NCNN_BENCHMARK && defined(NCNN_WEIGHT_TEST_INTERNAL)
    const uint64_t initial_capacity = weight_benchmark_counts()["device_capacity"];
#endif
    ncnn::VkMat result(256, (size_t)4u, &blobs);
    if (result.empty())
        return -1;
    std::vector<ncnn::VkMat> bindings(4);
    for (int i = 0; i < 3; i++)
        bindings[i] = prepared[i];
    bindings[3] = result;
    const std::vector<ncnn::vk_constant_type> constants;
    cmd.record_pipeline(&pipeline, bindings, constants, result);
    ncnn::Mat output;
    cmd.record_clone(result, output, opt);
    if (cmd.submit_and_wait())
        return -1;
    ncnn::Mat expected(256);
    for (int i = 0; i < 256; i++)
        ((float*)expected)[i] = ((const float*)sources[0])[i] + ((const float*)sources[1])[i] + ((const float*)sources[2])[i];
    if (CompareMat(expected, output, 0.0001f))
        return -1;
#if NCNN_BENCHMARK && defined(NCNN_WEIGHT_TEST_INTERNAL)
    if (weight_benchmark_counts()["device_capacity"] <= initial_capacity)
        return -1;
#endif
    // Retained cache contents remain usable after automatic growth.
    if (cmd.reset())
        return -1;
    cmd.record_clone(prepared[0], warm_output, opt);
    if (cmd.submit_and_wait() || CompareMat(sources[0], warm_output, 0.f))
        return -1;
    if (opt.weight_device_cache_size || opt.weight_host_cache_size || opt.weight_device_budget || opt.weight_host_budget)
        return -1;
    ncnn::Option fixed_opt = opt;
    fixed_opt.weight_device_cache_size = 2048;
    ncnn::VkCompute fixed(weights, fixed_opt);
    fixed.record_pipeline(&pipeline, bindings, constants, result);
    if (fixed.submit_and_wait() == 0)
        return -1;
    fprintf(stderr, "automatic weight cache growth, retained contents and fixed capacity verified\n");
    return 0;
}

static int load_scale_net(ncnn::Net& net, int storage, float scale)
{
    net.opt = weight_option();
    net.set_vulkan_device(gpu_index);
    net.opt.use_weights_in_host_memory = storage == 1;
    net.opt.use_weights_in_host_disk = storage >= 2;
    net.opt.weight_host_cache_size = 2048;
    net.opt.weight_device_cache_size = 4096;
    if (storage == 3)
    {
        net.opt.weight_device_budget = 65536;
        net.opt.weight_host_budget = 65536;
    }
    const char param[] = "7767517\n2 2\nInput input 0 1 input\nGemm scale 1 1 input output 5=1 6=1 8=16 9=16 10=-1\n";
    std::vector<unsigned char> model(4 + 16 * 16 * 4, 0);
    for (int i = 0; i < 16; i++)
        memcpy(model.data() + 4 + (i * 16 + i) * 4, &scale, 4);
    if (net.load_param_mem(param))
        return -1;
    const unsigned char* ptr = model.data();
    ncnn::DataReaderFromMemory reader(ptr);
    return net.load_model(reader);
}

class FailingOutputAllocator : public ncnn::Allocator
{
public:
    virtual void* fastMalloc(size_t)
    {
        return 0;
    }
    virtual void fastFree(void*)
    {
    }
};

static int test_net_load_failure()
{
    ncnn::Net net;
#if defined(__linux__)
    // Explicit directories are strict even without residency budgets, and survive clear/load_param.
    char invalid[] = "/dev/null/ncnn-weights";
    if (net.set_weight_file_directory(invalid))
        return -1;
    invalid[0] = 0;
    if (load_scale_net(net, 2, 1.f) == 0)
        return -1;
    net.clear();
    if (load_scale_net(net, 2, 1.f) == 0 || net.set_weight_file_directory("/tmp"))
        return -1;
#endif
    if (load_scale_net(net, 2, 1.f) || net.set_weight_file_directory(0) == 0)
        return -1;
    {
        FailingOutputAllocator allocator;
        ncnn::Extractor ex = net.create_extractor();
        ex.set_blob_allocator(&allocator);
        ncnn::Mat output;
        if (ex.input("input", RandomMat(16, 16)) || ex.extract("output", output) == 0)
            return -1;
    }
    net.clear();
    if (net.set_weight_file_directory(0))
        return -1;

    // The first layer uploads successfully before a later layer rejects truncated weights.
    const char param[] = "7767517\n3 3\nInput input 0 1 input\nGemm first 1 1 input hidden 5=1 6=1 8=16 9=16 10=-1\nGemm second 1 1 hidden output 5=1 6=1 8=16 9=16 10=-1\n";
    std::vector<unsigned char> model(4 + 16 * 16 * 4, 0);
    for (int storage = 0; storage < 3; storage++)
    {
        net.opt.use_weights_in_host_memory = storage == 1;
        net.opt.use_weights_in_host_disk = storage == 2;
        if (net.load_param_mem(param))
            return -1;
        const unsigned char* ptr = model.data();
        ncnn::DataReaderFromMemory reader(ptr, model.size());
        if (net.load_model(reader) == 0)
            return -1;
        net.clear();
        if (load_scale_net(net, storage, 1.f))
            return -1;
    }
    return 0;
}

static ncnn::Mat scaled_input(const ncnn::Mat& input, float scale)
{
    ncnn::Mat expected = input.clone();
    for (size_t i = 0; i < expected.total(); i++)
        ((float*)expected)[i] *= scale;
    return expected;
}

static void configure_external_extractor(ncnn::Extractor& ex, const ncnn::Option& opt)
{
    ex.set_blob_vkallocator(opt.blob_vkallocator);
    ex.set_workspace_vkallocator(opt.blob_vkallocator);
    ex.set_staging_vkallocator(opt.staging_vkallocator);
}

static int test_command_model_chain(int storage_a, int storage_b)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::Net a;
    ncnn::Net b;
    ncnn::Net device;
    if (load_scale_net(a, storage_a, 1.25f) || load_scale_net(b, storage_b, 0.5f) || load_scale_net(device, 0, 2.f))
        return -1;
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_host_cache_size = 2048;
    opt.weight_device_cache_size = 4096;
    ncnn::VkCompute cmd(vkdev, opt);
    ncnn::Mat input = RandomMat(16, 16);
    const ncnn::Mat expected = scaled_input(input, 1.25f * 0.5f * 1.25f);
#if defined(NCNN_WEIGHT_TEST_INTERNAL)
    reset_weight_benchmark();
#endif
    for (int round = 0; round < 3; round++)
    {
        if (round && cmd.reset())
            return -1;
        if (round == 2 && cmd.clear_weight_cache())
            return -1;
        ncnn::Extractor first = a.create_extractor();
        ncnn::Extractor middle = b.create_extractor();
        ncnn::Extractor last = a.create_extractor();
        configure_external_extractor(first, opt);
        configure_external_extractor(middle, opt);
        configure_external_extractor(last, opt);
        // These setters configure only an Extractor's internally owned command.
        first.set_weight_device_cache_size(16);
        middle.set_weight_device_cache_size(16);
        last.set_weight_device_cache_size(16);
        ncnn::VkMat gpu_input;
        ncnn::VkMat first_output;
        ncnn::VkMat middle_output;
        ncnn::VkMat last_output;
        cmd.record_upload(input, gpu_input, opt);
        if (first.input("input", gpu_input) || first.extract("output", first_output, cmd)
                || middle.input("input", first_output) || middle.extract("output", middle_output, cmd)
                || last.input("input", middle_output) || last.extract("output", last_output, cmd))
            return -1;
        if (round == 0 && cmd.clear_weight_cache() == 0)
            return -1;
        ncnn::Mat output;
        cmd.record_download(last_output, output, opt);
        if (cmd.submit_and_wait() || CompareMat(expected, output, 0.0001f))
            return -1;
#if NCNN_BENCHMARK && defined(NCNN_WEIGHT_TEST_INTERNAL)
        const uint64_t file_weights = (storage_a == 2 ? 1 : 0) + (storage_b == 2 ? 1 : 0);
        if (file_weights && weight_benchmark_counts()["file_reads"] != file_weights)
            return -1;
#endif
        if (round == 1)
        {
            // A retained cache must not turn unrelated DEVICE work into replay.
            if (cmd.reset())
                return -1;
#if defined(NCNN_WEIGHT_TEST_INTERNAL)
            reset_weight_benchmark();
#endif
            ncnn::Extractor direct = device.create_extractor();
            configure_external_extractor(direct, opt);
            ncnn::VkMat direct_input;
            ncnn::VkMat direct_output;
            ncnn::Mat downloaded;
            cmd.record_upload(input, direct_input, opt);
            if (direct.input("input", direct_input) || direct.extract("output", direct_output, cmd))
                return -1;
            cmd.record_download(direct_output, downloaded, opt);
            if (cmd.submit_and_wait() || CompareMat(scaled_input(input, 2.f), downloaded, 0.0001f))
                return -1;
#if NCNN_BENCHMARK && defined(NCNN_WEIGHT_TEST_INTERNAL)
            if (!weight_benchmark_counts().empty())
                return -1;
#endif
        }
    }
    return 0;
}

static int test_command_budget_scope()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator store(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    if (store.set_weights_in_host_disk(true) || store.set_weight_budgets(4096, 4096))
        return -1;
    ncnn::VkMat weights[3];
    ncnn::Mat source = RandomMat(512);
    for (int i = 0; i < 3; i++)
    {
        weights[i].create(512, (size_t)4u, &store);
        if (weights[i].empty() || !weights[i].data->external_ptr)
            return -1;
        memcpy(weights[i].data->external_ptr, source.data, 2048);
    }
    if (store.finalize_weights(2048, 2048))
        return -1;
    int managed = -1;
    for (int i = 0; i < 3; i++)
        if (weights[i].data->storage_type != 0)
            managed = i;
    if (managed < 0)
        return -1;
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_device_budget = 4096;
    opt.weight_host_budget = 4096;
    opt.weight_device_cache_size = 4096;
    opt.weight_host_cache_size = 2048;
    ncnn::VkCompute cmd(vkdev, opt);
    ncnn::Mat output;
    cmd.record_clone(weights[managed], output, opt);
    if (cmd.submit_and_wait() || CompareMat(source, output, 0.f) || cmd.reset() || cmd.clear_weight_cache())
        return -1;
    // The allocator-based constructor deliberately opts into that model's headroom.
    ncnn::VkCompute bounded(store, opt);
    bounded.record_clone(weights[managed], output, opt);
    if (bounded.submit_and_wait() == 0)
        return -1;
    return 0;
}

static int test_command_model_reload()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::Net net;
    ncnn::Net stable;
    if (load_scale_net(stable, 2, 0.75f))
        return -1;
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_device_cache_size = 4096;
    opt.weight_host_cache_size = 2048;
    ncnn::VkCompute cmd(vkdev, opt);
    ncnn::Mat input = RandomMat(16, 16);
#if defined(NCNN_WEIGHT_TEST_INTERNAL)
    reset_weight_benchmark();
#endif
    for (int round = 0; round < 8; round++)
    {
        const float scale = 1.f + round * 0.25f;
        if (load_scale_net(net, 2, scale))
            return -1;
        {
            ncnn::Extractor ex = net.create_extractor();
            ncnn::Extractor next = stable.create_extractor();
            configure_external_extractor(ex, opt);
            configure_external_extractor(next, opt);
            ncnn::VkMat gpu_input;
            ncnn::VkMat gpu_middle;
            ncnn::VkMat gpu_output;
            ncnn::Mat output;
            cmd.record_upload(input, gpu_input, opt);
            if (ex.input("input", gpu_input) || ex.extract("output", gpu_middle, cmd)
                    || next.input("input", gpu_middle) || next.extract("output", gpu_output, cmd))
                return -1;
            cmd.record_download(gpu_output, output, opt);
            if (cmd.submit_and_wait() || CompareMat(scaled_input(input, scale * 0.75f), output, 0.0001f) || cmd.reset())
                return -1;
        }
        net.clear();
#if NCNN_BENCHMARK && defined(NCNN_WEIGHT_TEST_INTERNAL)
        // Reloading one model must not invalidate the other model's cached copy.
        if (weight_benchmark_counts()["file_reads"] != (uint64_t)round + 2)
            return -1;
#endif
    }
    return cmd.clear_weight_cache();
}

static int test_recorded_weight_lifetime()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    ncnn::VkCompute cmd(vkdev, opt);
    ncnn::Mat original = RandomMat(128);
    ncnn::Mat output;
    for (int round = 0; round < 3; round++)
    {
        {
            ncnn::VkWeightAllocator store(vkdev);
            if (store.set_weights_in_host_disk(true))
                return -1;
            ncnn::VkMat weight(128, (size_t)4u, &store);
            if (weight.empty())
                return -1;
            memcpy(weight.data->external_ptr, original.data, original.total() * original.elemsize);
            if (store.finalize_weights())
                return -1;
            cmd.record_clone(weight, output, opt);
            if (round == 0)
            {
                weight.release();
                store.clear();
                if (cmd.submit_and_wait() == 0 || cmd.reset())
                    return -1;
            }
            if (round == 2 && (cmd.submit_and_wait() || CompareMat(original, output, 0.f) || cmd.reset()))
                return -1;
        }
        // Detect a destroyed allocator without reading the freed source buffer.
        if (round == 1 && (cmd.submit_and_wait() == 0 || cmd.reset()))
            return -1;
    }
    return 0;
}

class CountingWeightAllocator : public ncnn::VkBlobAllocator
{
public:
    explicit CountingWeightAllocator(const ncnn::VulkanDevice* vkdev)
        : ncnn::VkBlobAllocator(vkdev, 4096), allocations(0), frees(0)
    {
    }
    virtual ncnn::VkBufferMemory* fastMalloc(size_t size)
    {
        allocations++;
        return ncnn::VkBlobAllocator::fastMalloc(size);
    }
    virtual void fastFree(ncnn::VkBufferMemory* ptr)
    {
        frees++;
        ncnn::VkBlobAllocator::fastFree(ptr);
    }
    int allocations;
    int frees;
};

static int test_internal_command_pool()
{
    ncnn::Net net;
    if (load_scale_net(net, 2, 1.5f))
        return -1;
    ncnn::Mat input = RandomMat(16, 16);
    const ncnn::Mat expected = scaled_input(input, 1.5f);
#if defined(NCNN_WEIGHT_TEST_INTERNAL)
    reset_weight_benchmark();
#endif
    for (int round = 0; round < 3; round++)
    {
        ncnn::Extractor ex = net.create_extractor();
        ncnn::Mat output;
        if (ex.input("input", input) || ex.extract("output", output) || CompareMat(expected, output, 0.0001f))
            return -1;
#if NCNN_BENCHMARK && defined(NCNN_WEIGHT_TEST_INTERNAL)
        if (weight_benchmark_counts()["file_reads"] != 1 || weight_benchmark_counts()["cache_hits"] < (uint64_t)round)
            return -1;
#endif
    }
    // A caller allocator must not be borrowed from or retained by the idle pool.
    CountingWeightAllocator custom(ncnn::get_gpu_device(gpu_index));
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_weight_device_vkallocator(&custom);
        ncnn::Mat output;
        if (ex.input("input", input) || ex.extract("output", output) || CompareMat(expected, output, 0.0001f))
            return -1;
    }
    if (custom.allocations != 1 || custom.frees != 1)
        return -1;
    return 0;
}

static int test_default_command_options()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_index);
    ncnn::VkWeightAllocator store(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    if (store.set_weights_in_host_disk(true))
        return -1;
    ncnn::Mat input = RandomMat(128);
    ncnn::VkMat weight(128, (size_t)4u, &store);
    if (weight.empty())
        return -1;
    memcpy(weight.data->external_ptr, input.data, 128 * sizeof(float));
    if (store.finalize_weights())
        return -1;
    ncnn::VkCompute cmd(vkdev);
    for (int round = 0; round < 2; round++)
    {
        CountingWeightAllocator custom(vkdev);
        ncnn::Option opt = weight_option();
        opt.blob_vkallocator = &blobs;
        opt.staging_vkallocator = &staging;
        opt.weight_device_vkallocator = &custom;
        opt.weight_device_budget = round == 0 ? 0 : 1;
        ncnn::Mat output;
        cmd.record_clone(weight, output, opt);
        if (cmd.submit_and_wait() || CompareMat(input, output, 0.f) || custom.allocations || cmd.reset())
            return -1;
    }
    // A retained command must not depend on a clone caller's weight allocator.
    return cmd.clear_weight_cache();
}

static int test_internal_command_reload()
{
    ncnn::Net net;
    net.set_vulkan_device(gpu_index);
    net.opt = weight_option();
    net.opt.use_packing_layout = false;
    net.opt.use_weights_in_host_disk = true;
    net.opt.weight_device_budget = 8192;
    net.opt.weight_host_budget = 8192;
    const char* params[] = {
        "7767517\n3 3\nMemoryData w0 0 1 b0 0=1024\nMemoryData w1 0 1 b1 0=1024\nMemoryData w2 0 1 b2 0=1024\n",
        "7767517\n3 3\nMemoryData w0 0 1 b0 0=256\nMemoryData w1 0 1 b1 0=1536\nMemoryData w2 0 1 b2 0=1536\n"
    };
    ncnn::Mat model(3328);
    model.fill(1.f);
    const unsigned char* ptr = (const unsigned char*)model.data;
    ncnn::DataReaderFromMemory first_reader(ptr);
    if (net.load_param_mem(params[0]) || net.load_model(first_reader))
        return -1;
    ncnn::Extractor old = net.create_extractor();
    ncnn::Mat output;
    if (old.extract("b2", output) || old.input("b2", ncnn::Mat()) || old.input("b2", ncnn::VkMat()))
        return -1;
    output.release();
    net.clear();
    ptr = (const unsigned char*)model.data;
    ncnn::DataReaderFromMemory second_reader(ptr);
    if (net.load_param_mem(params[1]) || net.load_model(second_reader))
        return -1;
    // Reclaiming the old lease must not restore its smaller residency headroom.
    old.clear();
    ncnn::Extractor current = net.create_extractor();
    if (current.extract("b2", output) || output.total() != 1536)
        return -1;
    for (int i = 0; i < 1536; i++)
        if (output[i] != 1.f)
            return -1;
    return 0;
}

static int test_weight_execution(int argc, char** argv)
{
    SRAND(7767517);
#if NCNN_BENCHMARK && defined(NCNN_WEIGHT_TEST_INTERNAL)
    reset_weight_benchmark();
#endif
    if (ncnn::get_gpu_count() == 0)
        return 0;
    gpu_index = argc > 1 ? atoi(argv[1]) : ncnn::get_default_gpu_index();
    if (gpu_index < 0 || gpu_index >= ncnn::get_gpu_count())
        return -1;
    if (test_network() || test_net_load_failure() || test_empty_upload() || test_standalone_sources() || test_existing_model())
        return -1;
    for (int host = 0; host < 2; host++)
        for (int small_cache = 0; small_cache < 2; small_cache++)
            if (test_mixed_bindings(host != 0, small_cache != 0))
            {
                fprintf(stderr, "mixed weight bindings failed host=%d small_cache=%d\n", host, small_cache);
                return -1;
            }
    const int chains[][2] = {{2, 2}, {0, 2}, {2, 0}, {1, 2}, {2, 3}, {3, 3}, {1, 1}};
    for (size_t i = 0; i < sizeof(chains) / sizeof(chains[0]); i++)
    {
        if (test_command_model_chain(chains[i][0], chains[i][1]))
        {
            fprintf(stderr, "cross-model command chain failed storage=%d,%d\n", chains[i][0], chains[i][1]);
            return -1;
        }
    }
    if (test_default_command_options() || test_internal_command_reload() || test_command_budget_scope() || test_command_model_reload() || test_recorded_weight_lifetime() || test_internal_command_pool())
    {
        fprintf(stderr, "command source lifetime or context pool test failed\n");
        return -1;
    }
    if (test_automatic_weight_cache())
    {
        fprintf(stderr, "automatic weight cache test failed\n");
        return -1;
    }
#if defined(NCNN_WEIGHT_TEST_INTERNAL)
    if (test_copy_residency())
    {
        fprintf(stderr, "copy residency test failed\n");
        return -1;
    }
    for (int mode = 0; mode < 3; mode++)
    {
        if (test_cache_reuse(mode) || test_queue_ring(mode))
        {
            fprintf(stderr, "cache or command ring test failed mode=%d\n", mode);
            return -1;
        }
    }
    if (test_host_queue_modes(false) || test_host_queue_modes(true) || test_execution_failures())
    {
        fprintf(stderr, "host queue or failure recovery test failed\n");
        return -1;
    }
#endif
    const ncnn::GpuInfo& info = ncnn::get_gpu_info(gpu_index);
    for (precision = 1; precision <= 2; precision++)
    {
        if (precision == 1 && !info.support_fp16_storage())
            continue;
        if (precision == 2 && !info.support_bf16_storage() && !info.support_bf16_packed())
            continue;
        if (test_network())
            return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    const int ret = test_weight_execution(argc, argv);
    ncnn::destroy_gpu_instance();
    return ret;
}
