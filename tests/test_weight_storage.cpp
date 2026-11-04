// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "command.h"
#include "datareader.h"
#include "gpu.h"
#include "net.h"
#include "testutil.h"

#include <string>

// gemm chain with distinct per-layer weights and a memorydata bias
// input b0 and output out are both w=16 h=8
static int load_weight_storage_net(ncnn::Net& net, const ncnn::Option& opt, int seed, const char* directory = 0)
{
    const int layer_count = 6;

    net.opt = opt;
    net.set_weight_file_directory(directory);

    char line[256];
    sprintf(line, "7767517\n%d %d\nInput input 0 1 b0\n", layer_count + 3, layer_count + 3);
    std::string param = line;

    std::vector<float> model;
    int width = 16;
    for (int i = 0; i < layer_count; i++)
    {
        const int next_width = 16 + (i + 1) % 3 * 4;

        sprintf(line, "Gemm gemm%d 1 1 b%d b%d 5=1 6=1 8=%d 9=%d 10=-1\n", i, i, i + 1, next_width, width);
        param += line;

        // zero tag for raw float32, then K x N weight
        model.push_back(0.f);
        for (int y = 0; y < width; y++)
        {
            for (int x = 0; x < next_width; x++)
            {
                float v = x == y ? 1.f : 0.f;
                v += ((x * 3 + y * 7 + i * 11 + seed * 5) % 13 - 6) * 0.00390625f;
                if (x == (y + i + seed + 1) % next_width)
                    v += 0.125f;
                model.push_back(v);
            }
        }

        width = next_width;
    }

    sprintf(line, "MemoryData bias 0 1 bias 0=16 1=8\nBinaryOp add 2 1 b%d bias out 0=0\n", layer_count);
    param += line;

    for (int i = 0; i < 16 * 8; i++)
    {
        model.push_back(((i * 5 + seed * 3) % 17 - 8) * 0.0625f);
    }

    int ret = net.load_param_mem(param.c_str());
    if (ret != 0)
        return ret;

    const unsigned char* mem = (const unsigned char*)model.data();
    ncnn::DataReaderFromMemory dr(mem, model.size() * sizeof(float));
    return net.load_model(dr);
}

// disk weights require a usable temporary file
static bool support_weight_file()
{
    ncnn::TemporaryFile file;
    return file.open() == 0;
}

static ncnn::Option weight_storage_option(int precision)
{
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = precision == 1;
    opt.use_fp16_storage = precision == 1;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = precision == 2;
    opt.use_bf16_storage = precision == 2;
    return opt;
}

static int extract_weight_storage_net(const ncnn::Net& net, const ncnn::Mat& input, ncnn::Mat& output)
{
    ncnn::Extractor ex = net.create_extractor();

    int ret = ex.input("b0", input);
    if (ret != 0)
        return ret;

    return ex.extract("out", output);
}

static int extract_weight_storage_reference(int seed, const ncnn::Mat& input, ncnn::Mat& output)
{
    ncnn::Option opt = weight_storage_option(0);
    opt.use_vulkan_compute = false;

    ncnn::Net net;
    int ret = load_weight_storage_net(net, opt, seed);
    if (ret != 0)
        return ret;

    return extract_weight_storage_net(net, input, output);
}

#if NCNN_THREADS
struct weight_storage_task
{
    const ncnn::Net* net;
    const ncnn::Mat* input;
    ncnn::Mat output;
    int ret;
};

static void* weight_storage_thread(void* args)
{
    weight_storage_task* task = (weight_storage_task*)args;
    task->ret = extract_weight_storage_net(*task->net, *task->input, task->output);
    return 0;
}
#endif // NCNN_THREADS

// record the net into an external command, as zero-copy pipelines do
static int extract_weight_storage_command(const ncnn::Net& net, ncnn::VkCompute& cmd, const ncnn::Mat& input, ncnn::Mat& output)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    int ret = 0;
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_blob_vkallocator(blob_vkallocator);
        ex.set_workspace_vkallocator(blob_vkallocator);
        ex.set_staging_vkallocator(staging_vkallocator);

        ncnn::VkMat input_gpu;
        cmd.record_upload(input, input_gpu, opt);

        ncnn::VkMat output_gpu;
        ex.input("b0", input_gpu);
        ex.extract("out", output_gpu, cmd);

        cmd.record_download(output_gpu, output, opt);

        ret = cmd.submit_and_wait();
        if (ret == 0)
            ret = cmd.reset();
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return ret;
}

static int test_weight_storage(const ncnn::Option& opt, float epsilon)
{
    const int seed = 3;

    ncnn::Mat input = RandomMat(16, 8);

    ncnn::Mat expected;
    if (extract_weight_storage_reference(seed, input, expected) != 0)
    {
        fprintf(stderr, "test_weight_storage reference failed\n");
        return -1;
    }

    ncnn::Net net;
    if (load_weight_storage_net(net, opt, seed) != 0)
    {
        fprintf(stderr, "test_weight_storage load failed\n");
        return -1;
    }

    // repeated extraction reuses the stored weights
    for (int i = 0; i < 3; i++)
    {
        ncnn::Mat output;
        if (extract_weight_storage_net(net, input, output) != 0 || CompareMat(expected, output, epsilon) != 0)
        {
            fprintf(stderr, "test_weight_storage extract failed round=%d\n", i);
            return -1;
        }
    }

    {
        ncnn::VkCompute cmd(net.vulkan_device());
        cmd.set_use_weight_upload_online(opt.use_weights_in_host_memory || opt.use_weights_in_disk);

        for (int i = 0; i < 2; i++)
        {
            ncnn::Mat output;
            if (extract_weight_storage_command(net, cmd, input, output) != 0 || CompareMat(expected, output, epsilon) != 0)
            {
                fprintf(stderr, "test_weight_storage command failed round=%d\n", i);
                return -1;
            }
        }
    }

#if NCNN_THREADS
    weight_storage_task tasks[3];
    ncnn::Thread* threads[3];
    for (int i = 0; i < 3; i++)
    {
        tasks[i].net = &net;
        tasks[i].input = &input;
        tasks[i].ret = -1;
        threads[i] = new ncnn::Thread(weight_storage_thread, &tasks[i]);
    }
    for (int i = 0; i < 3; i++)
    {
        threads[i]->join();
        delete threads[i];
    }
    for (int i = 0; i < 3; i++)
    {
        if (tasks[i].ret != 0 || CompareMat(expected, tasks[i].output, epsilon) != 0)
        {
            fprintf(stderr, "test_weight_storage concurrent extract failed thread=%d\n", i);
            return -1;
        }
    }
#endif // NCNN_THREADS

    return 0;
}

static int test_weight_storage_0()
{
    // host disk device_limit host_limit
    const size_t storages[][4] = {
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 1, 4096, 4096},
        {0, 1, 1048576, 1048576},
    };

    const int storage_count = sizeof(storages) / sizeof(storages[0]);

    for (int precision = 0; precision < 3; precision++)
    {
        for (int i = 0; i < storage_count; i++)
        {
            if (storages[i][2] && !support_weight_file())
                continue;

            ncnn::Option opt = weight_storage_option(precision);
            opt.use_weights_in_host_memory = storages[i][0];
            opt.use_weights_in_disk = storages[i][1];
            opt.weight_device_memory_limit = storages[i][2];
            opt.weight_host_memory_limit = storages[i][3];

            const float epsilon = precision == 0 ? 0.001f : 0.05f;

            int ret = test_weight_storage(opt, epsilon);
            if (ret != 0)
            {
                fprintf(stderr, "test_weight_storage failed precision=%d host=%d disk=%d device_limit=%d host_limit=%d\n", precision, (int)storages[i][0], (int)storages[i][1], (int)storages[i][2], (int)storages[i][3]);
                return ret;
            }
        }
    }

    return 0;
}

// clear and load other weights into the same net
static int test_weight_storage_1()
{
    ncnn::Option opt = weight_storage_option(0);
    opt.use_weights_in_disk = true;
    opt.weight_device_memory_limit = 4096;
    opt.weight_host_memory_limit = 4096;

    ncnn::Mat input = RandomMat(16, 8);

    ncnn::Net net;
    for (int seed = 0; seed < 3; seed++)
    {
        net.clear();

        ncnn::Mat expected;
        if (extract_weight_storage_reference(seed, input, expected) != 0)
            return -1;

        ncnn::Mat output;
        if (load_weight_storage_net(net, opt, seed) != 0 || extract_weight_storage_net(net, input, output) != 0 || CompareMat(expected, output, 0.001f) != 0)
        {
            fprintf(stderr, "test_weight_storage_1 failed seed=%d\n", seed);
            return -1;
        }
    }

    return 0;
}

static int test_weight_storage_chain(const ncnn::Option& opt0, const ncnn::Option& opt1)
{
    ncnn::Mat input = RandomMat(16, 8);

    ncnn::Mat middle;
    ncnn::Mat expected;
    if (extract_weight_storage_reference(0, input, middle) != 0 || extract_weight_storage_reference(1, middle, expected) != 0)
        return -1;

    ncnn::Net net0;
    ncnn::Net net1;
    if (load_weight_storage_net(net0, opt0, 0) != 0 || load_weight_storage_net(net1, opt1, 1) != 0)
        return -1;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = opt0;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    ncnn::Mat output;
    int ret = 0;
    {
        // two nets with different weight storage share one command
        ncnn::VkCompute cmd(vkdev);
        cmd.set_use_weight_upload_online(true);

        ncnn::Extractor ex0 = net0.create_extractor();
        ex0.set_blob_vkallocator(blob_vkallocator);
        ex0.set_workspace_vkallocator(blob_vkallocator);
        ex0.set_staging_vkallocator(staging_vkallocator);

        ncnn::Extractor ex1 = net1.create_extractor();
        ex1.set_blob_vkallocator(blob_vkallocator);
        ex1.set_workspace_vkallocator(blob_vkallocator);
        ex1.set_staging_vkallocator(staging_vkallocator);

        ncnn::VkMat input_gpu;
        cmd.record_upload(input, input_gpu, opt);

        ncnn::VkMat middle_gpu;
        ex0.input("b0", input_gpu);
        ex0.extract("out", middle_gpu, cmd);

        ncnn::VkMat output_gpu;
        ex1.input("b0", middle_gpu);
        ex1.extract("out", output_gpu, cmd);

        cmd.record_download(output_gpu, output, opt);

        ret = cmd.submit_and_wait();
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    if (ret != 0 || CompareMat(expected, output, 0.001f) != 0)
    {
        fprintf(stderr, "test_weight_storage_chain failed host=%d,%d disk=%d,%d\n", opt0.use_weights_in_host_memory, opt1.use_weights_in_host_memory, opt0.use_weights_in_disk, opt1.use_weights_in_disk);
        return -1;
    }

    return 0;
}

static int test_weight_storage_2()
{
    ncnn::Option device = weight_storage_option(0);

    ncnn::Option host = device;
    host.use_weights_in_host_memory = true;

    ncnn::Option disk = device;
    disk.use_weights_in_disk = true;

    ncnn::Option mixed = disk;
    mixed.weight_device_memory_limit = 4096;
    mixed.weight_host_memory_limit = 4096;

    return 0
           || test_weight_storage_chain(device, disk)
           || test_weight_storage_chain(disk, host)
           || test_weight_storage_chain(mixed, disk);
}

// weights uploaded and consumed through the allocator api
static int test_weight_storage_allocator(size_t device_memory_limit, size_t host_memory_limit)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;

    const int sizes[4] = {7, 256, 1000, 4096};

    ncnn::Mat inputs[4];
    for (int i = 0; i < 4; i++)
    {
        inputs[i] = RandomMat(sizes[i]);
    }

    ncnn::VkWeightAllocator weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator weight_staging_vkallocator(vkdev);

    weight_vkallocator.set_weights_in_disk(true);
    if (weight_vkallocator.set_weight_memory_limit(device_memory_limit, host_memory_limit) != 0)
    {
        fprintf(stderr, "set_weight_memory_limit failed\n");
        return -1;
    }

    ncnn::Option opt = weight_storage_option(0);
    opt.blob_vkallocator = &weight_vkallocator;
    opt.workspace_vkallocator = &weight_vkallocator;
    opt.staging_vkallocator = &weight_staging_vkallocator;

    ncnn::VkMat weights[4];
    {
        ncnn::VkTransfer cmd(vkdev);
        for (int i = 0; i < 4; i++)
        {
            cmd.record_upload(inputs[i], weights[i], opt);
        }

        if (cmd.submit_and_wait() != 0 || weight_vkallocator.finalize_weights() != 0)
        {
            fprintf(stderr, "weight upload failed\n");
            return -1;
        }
    }

    // resident weights stay within the limits
    uint64_t device_bytes = 0;
    uint64_t host_bytes = 0;
    weight_vkallocator.get_weight_memory_usage(device_bytes, host_bytes);

    bool exceeded = device_bytes > device_memory_limit || host_bytes > host_memory_limit;
    if (unified)
    {
        // integrated gpu shares one limit
        exceeded = device_bytes + host_bytes > std::min(device_memory_limit, host_memory_limit);
    }
    if (exceeded)
    {
        fprintf(stderr, "weight memory usage %lu %lu exceeds limit %lu %lu\n", (unsigned long)device_bytes, (unsigned long)host_bytes, (unsigned long)device_memory_limit, (unsigned long)host_memory_limit);
        return -1;
    }

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    ncnn::Mat outputs[4];
    ncnn::Mat output_copy;
    int ret = 0;
    {
        ncnn::VkCompute cmd(vkdev);
        cmd.set_use_weight_upload_online(true);

        for (int i = 0; i < 4; i++)
        {
            cmd.record_clone(weights[i], outputs[i], opt);
        }

        ncnn::VkMat weight_copy;
        cmd.record_clone(weights[3], weight_copy, opt);
        cmd.record_download(weight_copy, output_copy, opt);

        ret = cmd.submit_and_wait();
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    if (ret != 0)
    {
        fprintf(stderr, "weight download failed\n");
        return -1;
    }

    for (int i = 0; i < 4; i++)
    {
        if (CompareMat(inputs[i], outputs[i], 0.f) != 0)
        {
            fprintf(stderr, "weight value mismatch size=%d\n", sizes[i]);
            return -1;
        }
    }

    if (CompareMat(inputs[3], output_copy, 0.f) != 0)
    {
        fprintf(stderr, "weight copy value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_weight_storage_3()
{
    return 0
           || test_weight_storage_allocator(0, 0)
           || test_weight_storage_allocator(4096, 4096)
           || test_weight_storage_allocator(1048576, 1048576);
}

static int test_weight_storage_4()
{
    // an explicit weight file directory must be writable
    {
        ncnn::Option opt = weight_storage_option(0);
        opt.use_weights_in_disk = true;

        ncnn::Net net;
        if (load_weight_storage_net(net, opt, 0, "ncnn-weight-storage-missing/weights") == 0)
        {
            fprintf(stderr, "test_weight_storage_4 unwritable weight file directory accepted\n");
            return -1;
        }
    }

    // discrete gpu requires both device and host limits
    {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info();
        const bool unified = info.type() == 1 || info.type() == 3;

        ncnn::Option opt = weight_storage_option(0);
        opt.use_weights_in_disk = true;
        opt.weight_device_memory_limit = 4096;
        opt.weight_host_memory_limit = 0;

        ncnn::Net net;
        int ret = load_weight_storage_net(net, opt, 0);
        if (unified && ret != 0)
        {
            fprintf(stderr, "test_weight_storage_4 single weight memory limit rejected on unified memory\n");
            return -1;
        }
        if (!unified && ret == 0)
        {
            fprintf(stderr, "test_weight_storage_4 single weight memory limit accepted on discrete memory\n");
            return -1;
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    if (ncnn::get_gpu_count() == 0)
        return 0;

    if (!support_weight_file())
        return test_weight_storage_0();

    return 0
           || test_weight_storage_0()
           || test_weight_storage_1()
           || test_weight_storage_2()
           || test_weight_storage_3()
           || test_weight_storage_4();
}
