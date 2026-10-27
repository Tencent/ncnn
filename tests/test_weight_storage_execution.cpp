// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_weight_storage.h"

struct extract_task
{
    const ncnn::Net* net;
    const ncnn::Mat* input;
    ncnn::Mat output;
    int ret;
};

static void* extract_thread(void* args)
{
    extract_task* task = (extract_task*)args;
    ncnn::Extractor ex = task->net->create_extractor();
    task->ret = ex.input("b0", *task->input);
    if (task->ret == 0)
        task->ret = ex.extract("b12", task->output);
    return 0;
}

static int test_weight_storage_network(const ncnn::Option& opt)
{
    ncnn::Net net;
    if (load_weight_storage_net(net, opt, 1.f, 12, true) != 0)
    {
        fprintf(stderr, "test_weight_storage_network load failed host=%d disk=%d fp16=%d bf16=%d\n", opt.use_weights_in_host_memory, opt.use_weights_in_disk, opt.use_fp16_storage, opt.use_bf16_storage);
        return -1;
    }

    ncnn::Mat input = RandomMat(16, 8);
    ncnn::Net reference;
    ncnn::Option cpu_opt = weight_storage_option();
    cpu_opt.use_vulkan_compute = false;
    if (load_weight_storage_net(reference, cpu_opt, 1.f, 12, true) != 0)
        return -1;
    ncnn::Mat expected;
    ncnn::Extractor cpu = reference.create_extractor();
    if (cpu.input("b0", input) != 0 || cpu.extract("b12", expected) != 0)
        return -1;
    for (int i = 0; i < 3; i++)
    {
        ncnn::Extractor ex = net.create_extractor();
        ncnn::Mat output;
        if (ex.input("b0", input) != 0 || ex.extract("b12", output) != 0 || CompareMat(expected, output, 0.02f) != 0)
        {
            fprintf(stderr, "test_weight_storage_network extract failed round=%d host=%d disk=%d\n", i, opt.use_weights_in_host_memory, opt.use_weights_in_disk);
            return -1;
        }
    }

#if NCNN_THREADS
    extract_task tasks[3];
    ncnn::Thread* threads[3];
    for (int i = 0; i < 3; i++)
    {
        tasks[i].net = &net;
        tasks[i].input = &input;
        tasks[i].ret = 0;
        threads[i] = new ncnn::Thread(extract_thread, &tasks[i]);
    }
    for (int i = 0; i < 3; i++)
    {
        threads[i]->join();
        delete threads[i];
    }
    for (int i = 0; i < 3; i++)
    {
        if (tasks[i].ret != 0 || CompareMat(expected, tasks[i].output, 0.02f) != 0)
        {
            fprintf(stderr, "test_weight_storage_network concurrent extract failed thread=%d\n", i);
            return -1;
        }
    }
#endif // NCNN_THREADS
    return 0;
}

static int test_weight_storage_networks()
{
    ncnn::Option opt = weight_storage_option();
    if (test_weight_storage_network(opt) != 0)
        return -1;
    opt.use_weights_in_host_memory = true;
    if (test_weight_storage_network(opt) != 0)
        return -1;
    opt.use_weights_in_host_memory = false;
    opt.use_weights_in_disk = true;
    opt.weight_device_cache_size = 2048;
    opt.weight_host_cache_size = 256;
    if (test_weight_storage_network(opt) != 0)
        return -1;
    opt.weight_device_memory_limit = 6144;
    opt.weight_host_memory_limit = 6144;
    if (test_weight_storage_network(opt) != 0)
        return -1;
    opt.weight_device_memory_limit = 65536;
    opt.weight_host_memory_limit = 65536;
    if (test_weight_storage_network(opt) != 0)
        return -1;

    const ncnn::GpuInfo& info = ncnn::get_gpu_info(ncnn::get_default_gpu_index());
    // The half precision model exceeds both residency budgets together.
    opt.weight_device_memory_limit = 2048;
    opt.weight_host_memory_limit = 2048;
    opt.weight_device_cache_size = 1024;
    if (info.support_fp16_storage())
    {
        opt.use_fp16_packed = true;
        opt.use_fp16_storage = true;
        if (test_weight_storage_network(opt) != 0)
            return -1;
    }
    if (info.support_bf16_storage())
    {
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_bf16_packed = true;
        opt.use_bf16_storage = true;
        if (test_weight_storage_network(opt) != 0)
            return -1;
    }
    return 0;
}

static int test_weight_storage_model_chain(const ncnn::Option& first_opt, const ncnn::Option& second_opt)
{
    ncnn::Net first;
    ncnn::Net second;
    if (load_weight_storage_net(first, first_opt, 1.5f, 1) != 0 || load_weight_storage_net(second, second_opt, 0.5f, 1) != 0)
    {
        fprintf(stderr, "test_weight_storage_model_chain load failed\n");
        return -1;
    }

    const ncnn::VulkanDevice* vkdev = first.vulkan_device();
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_device_cache_size = 4096;
    opt.weight_host_cache_size = 256;
    ncnn::VkCompute cmd(vkdev, opt);
    ncnn::Mat input = RandomMat(16, 8);
    for (int i = 0; i < 3; i++)
    {
        ncnn::Extractor ex = first.create_extractor();
        ncnn::Extractor next = second.create_extractor();
        configure_weight_storage_extractor(ex, opt);
        configure_weight_storage_extractor(next, opt);
        ncnn::VkMat in;
        ncnn::VkMat middle;
        ncnn::VkMat out;
        ncnn::Mat output;
        cmd.record_upload(input, in, opt);
        if (ex.input("b0", in) != 0 || ex.extract("b1", middle, cmd) != 0 || next.input("b0", middle) != 0 || next.extract("b1", out, cmd) != 0)
        {
            fprintf(stderr, "test_weight_storage_model_chain record failed\n");
            return -1;
        }
        cmd.record_download(out, output, opt);
        if (cmd.submit_and_wait() != 0 || CompareMat(scale_weight_storage_input(input, 0.75f), output, 0.0001f) != 0 || cmd.reset() != 0)
        {
            fprintf(stderr, "test_weight_storage_model_chain output failed round=%d\n", i);
            return -1;
        }
    }
    if (cmd.clear_weight_cache() != 0)
    {
        fprintf(stderr, "test_weight_storage_model_chain clear failed\n");
        return -1;
    }
    return 0;
}

static int test_weight_storage_model_chains()
{
    ncnn::Option device = weight_storage_option();
    ncnn::Option host = device;
    host.use_weights_in_host_memory = true;
    ncnn::Option disk = device;
    disk.use_weights_in_disk = true;
    ncnn::Option resident = disk;
    resident.weight_device_memory_limit = 65536;
    resident.weight_host_memory_limit = 65536;
    return 0
           || test_weight_storage_model_chain(disk, disk)
           || test_weight_storage_model_chain(device, disk)
           || test_weight_storage_model_chain(disk, device)
           || test_weight_storage_model_chain(host, disk)
           || test_weight_storage_model_chain(disk, resident)
           || test_weight_storage_model_chain(resident, resident);
}

static int test_weight_storage_model_reload()
{
    ncnn::Option opt = weight_storage_option();
    opt.use_weights_in_disk = true;
    ncnn::Net net;
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option command_opt = opt;
    command_opt.blob_vkallocator = &blobs;
    command_opt.staging_vkallocator = &staging;
    ncnn::VkCompute cmd(vkdev, command_opt);
    ncnn::Mat input = RandomMat(16, 8);
    for (int i = 0; i < 5; i++)
    {
        float scale = 1.f + i * 0.25f;
        if (load_weight_storage_net(net, opt, scale, 1) != 0)
        {
            fprintf(stderr, "test_weight_storage_model_reload load failed round=%d\n", i);
            return -1;
        }
        {
            ncnn::Extractor ex = net.create_extractor();
            configure_weight_storage_extractor(ex, command_opt);
            ncnn::VkMat in;
            ncnn::VkMat out;
            ncnn::Mat output;
            cmd.record_upload(input, in, command_opt);
            if (ex.input("b0", in) != 0 || ex.extract("b1", out, cmd) != 0)
            {
                fprintf(stderr, "test_weight_storage_model_reload record failed round=%d\n", i);
                return -1;
            }
            cmd.record_download(out, output, command_opt);
            if (cmd.submit_and_wait() != 0 || CompareMat(scale_weight_storage_input(input, scale), output, 0.0001f) != 0 || cmd.reset() != 0)
            {
                fprintf(stderr, "test_weight_storage_model_reload output failed round=%d\n", i);
                return -1;
            }
        }
        net.clear();
    }
    return 0;
}

static int test_weight_storage_load_failure()
{
    ncnn::Option opt = weight_storage_option();
    opt.use_weights_in_disk = true;
    ncnn::Net net;
    char path[] = "ncnn-weight-directory-that-does-not-exist/weights";
    net.set_weight_file_directory(path);
    path[0] = 0;
    if (load_weight_storage_net(net, opt, 1.f, 1) == 0)
    {
        fprintf(stderr, "test_weight_storage_load_failure invalid directory accepted\n");
        return -1;
    }
    net.clear();
    net.set_weight_file_directory((const char*)0);
    if (load_weight_storage_net(net, opt, 1.f, 1) != 0)
    {
        fprintf(stderr, "test_weight_storage_load_failure reload failed\n");
        return -1;
    }
    net.clear();

    opt.weight_device_memory_limit = 256;
    opt.weight_host_memory_limit = 256;
    if (load_weight_storage_net(net, opt, 1.f, 12) == 0)
    {
        fprintf(stderr, "test_weight_storage_load_failure insufficient memory limit accepted\n");
        return -1;
    }
    net.clear();
    opt.weight_device_memory_limit = 0;
    opt.weight_host_memory_limit = 0;
    net.opt = opt;
    const char param[] = "7767517\n3 3\nInput input 0 1 b0\nGemm first 1 1 b0 b1 5=1 6=1 8=16 9=16 10=-1\nGemm second 1 1 b1 b2 5=1 6=1 8=16 9=16 10=-1\n";
    std::vector<unsigned char> model(4 + 16 * 16 * 4, 0);
    const unsigned char* ptr = model.data();
    ncnn::DataReaderFromMemory reader(ptr, model.size());
    if (net.load_param_mem(param) != 0 || net.load_model(reader) == 0)
    {
        fprintf(stderr, "test_weight_storage_load_failure truncated model accepted\n");
        return -1;
    }
    net.clear();
    if (load_weight_storage_net(net, opt, 1.f, 1) != 0)
    {
        fprintf(stderr, "test_weight_storage_load_failure recovery failed\n");
        return -1;
    }
    return 0;
}

static int test_weight_storage_internal_command_reload()
{
    ncnn::Net net;
    net.set_vulkan_device(ncnn::get_default_gpu_index());
    net.opt = weight_storage_option();
    net.opt.use_packing_layout = false;
    net.opt.use_weights_in_disk = true;
    net.opt.weight_device_memory_limit = 8192;
    net.opt.weight_host_memory_limit = 8192;
    const char* params[] = {
        "7767517\n3 3\nMemoryData w0 0 1 b0 0=1024\nMemoryData w1 0 1 b1 0=1024\nMemoryData w2 0 1 b2 0=1024\n",
        "7767517\n3 3\nMemoryData w0 0 1 b0 0=256\nMemoryData w1 0 1 b1 0=1536\nMemoryData w2 0 1 b2 0=1536\n"
    };
    ncnn::Mat model(3328);
    model.fill(1.f);
    const unsigned char* ptr = (const unsigned char*)model.data;
    ncnn::DataReaderFromMemory first_reader(ptr);
    if (net.load_param_mem(params[0]) || net.load_model(first_reader))
    {
        fprintf(stderr, "test_weight_storage_internal_command_reload failed at line %d\n", __LINE__);
        return -1;
    }
    ncnn::Extractor old = net.create_extractor();
    ncnn::Mat output;
    if (old.extract("b2", output) || old.input("b2", ncnn::Mat()) || old.input("b2", ncnn::VkMat()))
    {
        fprintf(stderr, "test_weight_storage_internal_command_reload failed at line %d\n", __LINE__);
        return -1;
    }
    output.release();
    net.clear();
    ptr = (const unsigned char*)model.data;
    ncnn::DataReaderFromMemory second_reader(ptr);
    if (net.load_param_mem(params[1]) || net.load_model(second_reader))
    {
        fprintf(stderr, "test_weight_storage_internal_command_reload failed at line %d\n", __LINE__);
        return -1;
    }
    // Reclaiming the old lease must not restore its smaller residency headroom.
    old.clear();
    ncnn::Extractor current = net.create_extractor();
    if (current.extract("b2", output) || output.total() != 1536)
    {
        fprintf(stderr, "test_weight_storage_internal_command_reload failed at line %d\n", __LINE__);
        return -1;
    }
    for (int i = 0; i < 1536; i++)
        if (output[i] != 1.f)
        {
            fprintf(stderr, "test_weight_storage_internal_command_reload failed at line %d\n", __LINE__);
            return -1;
        }
    return 0;
}

static int test_weight_storage_net_cache_limit()
{
    ncnn::Option opt = weight_storage_option();
    opt.use_weights_in_disk = true;
    opt.weight_device_cache_size = 256;
    opt.weight_host_cache_size = 256;
    ncnn::Net net;
    if (load_weight_storage_net(net, opt, 1.f, 1) != 0)
        return -1;
    ncnn::Extractor ex = net.create_extractor();
    ncnn::Mat output;
    if (ex.input("b0", RandomMat(16, 8)) != 0 || ex.extract("b1", output) == 0)
    {
        fprintf(stderr, "test_weight_storage_net_cache_limit ignored explicit cache size\n");
        return -1;
    }
    return 0;
}

int main()
{
    SRAND(7767517);
    if (ncnn::get_gpu_count() == 0)
        return 0;

    return 0
           || test_weight_storage_net_cache_limit()
           || test_weight_storage_networks()
           || test_weight_storage_model_chains()
           || test_weight_storage_model_reload()
           || test_weight_storage_load_failure()
           || test_weight_storage_internal_command_reload();
}
