// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"
#include "testutil.h"

#include <stdio.h>
#include <string.h>

#if NCNN_THREADS
struct GpuInstanceThreadContext
{
    ncnn::Mutex mutex;
    ncnn::ConditionVariable condition;
    bool start;
    int ret;
};

static void* create_gpu_instance_thread(void* args)
{
    GpuInstanceThreadContext* context = (GpuInstanceThreadContext*)args;
    {
        ncnn::MutexLockGuard lock(context->mutex);
        while (!context->start)
            context->condition.wait(context->mutex);
    }

    int ret = ncnn::create_gpu_instance();
    if (ret != 0)
    {
        ncnn::MutexLockGuard lock(context->mutex);
        context->ret = ret;
    }

    return 0;
}
#endif // NCNN_THREADS

static int test_gpu_instance()
{
    ncnn::destroy_gpu_instance();
    ncnn::destroy_gpu_instance();
    if (ncnn::get_gpu_instance())
    {
        fprintf(stderr, "test_gpu_instance failed to destroy instance\n");
        return -1;
    }

#if NCNN_THREADS
    GpuInstanceThreadContext context;
    context.start = false;
    context.ret = 0;

    const int thread_count = 24;
    ncnn::Thread* threads[thread_count];
    for (int i = 0; i < thread_count; i++)
        threads[i] = new ncnn::Thread(create_gpu_instance_thread, &context);

    {
        ncnn::MutexLockGuard lock(context.mutex);
        context.start = true;
        context.condition.broadcast();
    }

    for (int i = 0; i < thread_count; i++)
    {
        threads[i]->join();
        delete threads[i];
    }

    int ret = context.ret;
#else
    int ret = ncnn::create_gpu_instance();
#endif // NCNN_THREADS

    if (ret != 0 || !ncnn::get_gpu_instance() || ncnn::get_gpu_count() == 0)
    {
        fprintf(stderr, "test_gpu_instance failed to recreate instance ret=%d\n", ret);
        return -1;
    }

    return 0;
}

static int test_compile_spirv_module(const char* source, int expected, int comp_data_size = -1)
{
    const int source_size = strlen(source);
    char* comp_data = new char[source_size];
    memcpy(comp_data, source, source_size);

    if (comp_data_size == -1)
        comp_data_size = source_size;

    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;

    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(comp_data, comp_data_size, opt, spirv);
    delete[] comp_data;

    if (ret != expected)
    {
        fprintf(stderr, "test_compile_spirv_module failed ret=%d expected=%d comp_data_size=%d source=%s\n", ret, expected, comp_data_size, source);
        return -1;
    }

    return 0;
}

static int test_compile_spirv_module_0()
{
    return 0
           || test_compile_spirv_module("#version 450\nlayout(local_size_x = 1) in;\nvoid main() {}", 0)
           || test_compile_spirv_module("#version 450\nlayout(local_size_x = 1) in;\nvoid main() {}\n", 0)
           || test_compile_spirv_module("#version 450\r\nlayout(local_size_x = 1) in;\r\nvoid main() {}", 0)
           || test_compile_spirv_module("#version 450\rlayout(local_size_x = 1) in;\rvoid main() {}", 0)
           || test_compile_spirv_module("// leading comment\n#version\t450\n \t\r\nlayout(local_size_x = 1) in;\nvoid main() {}", 0)
           || test_compile_spirv_module("#version 450\n", 0)
           || test_compile_spirv_module("#version 450\n \t\r\n", 0)
           || test_compile_spirv_module("#version 450\n   #error outside source", 0, 13);
}

static int test_compile_spirv_module_1()
{
    return 0
           || test_compile_spirv_module("", -1)
           || test_compile_spirv_module("#version", -1)
           || test_compile_spirv_module("#version \t", -1)
           || test_compile_spirv_module("#version +", -1)
           || test_compile_spirv_module("#version nope\n", -1)
           || test_compile_spirv_module("void main() {}", -1)
           || test_compile_spirv_module("x#version 450\n", -1)
           || test_compile_spirv_module("#version 450\n#error test", -1)
           || test_compile_spirv_module("#version 450\n#error test\n", -1);
}

int main()
{
    SRAND(7767517);

    if (ncnn::get_gpu_count() == 0)
        return 0;

    return 0
           || test_gpu_instance()
           || test_compile_spirv_module_0()
           || test_compile_spirv_module_1();
}
