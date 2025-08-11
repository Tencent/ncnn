#include "benchmark.h"
#include "testutil.h"
#include "layer.h"
#include "mat.h"
#include "option.h"
#include "gpu.h"
#include "pipelinecache.h"

#include <future>
#include <stdio.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

void random_truncate_file(const char* filename, size_t new_size)
{
    FILE* fp = fopen(filename, "rb+");
    if (!fp) return;
#ifdef _WIN32
    int fd = _fileno(fp);
    _chsize(fd, new_size);
#else
    int fd = fileno(fp);
    ftruncate(fd, new_size);
#endif
    fclose(fp);
}

void corrupt_file(const char* filename)
{
    int mode = RandomInt(0, 10000) % 3;
    if (mode == 0)
    {
        if (remove(filename) != 0)
            fprintf(stderr, "Failed to remove file %s\n", filename);
        return;
    }
    if (mode == 1)
    {
        // empty file
        FILE* f = fopen(filename, "wb");
        if (!f) return;
        fclose(f);
        return;
    }
    // truncate to random size between 1 and original file size
    FILE* fp = fopen(filename, "rb");
    if (!fp) return;
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fclose(fp);

    size_t new_size = (size_t)(RandomInt(0, 10000) % file_size + 1);
    random_truncate_file(filename, new_size);
}

bool test_pipeline_creation(const ncnn::Option& opt, double* build_time = nullptr, int layer_type_index = 0)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);
    ncnn::Pipeline pipeline(vkdev);
    double start = ncnn::get_current_time();
    int ret = pipeline.create(0, opt, std::vector<ncnn::vk_specialization_type> {1});
    double end = ncnn::get_current_time();
    if (build_time) *build_time = end - start;
    if (ret != 0) return false;
    return true;
}

bool pipeline_cache_test_basic_creation()
{
    fprintf(stdout, "Start basic test\n");
    ncnn::create_gpu_instance();
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);
    const int options[][6] = {
        {0, 0, 0, 0, 0, 0},
    };

    ncnn::Option opt{};
    opt.num_threads = 1;
    opt.use_packing_layout = options[0][0];
    opt.use_fp16_packed = options[0][1];
    opt.use_fp16_storage = options[0][2];
    opt.use_fp16_arithmetic = options[0][3];
    opt.use_bf16_storage = options[0][4];
    opt.use_shader_pack8 = options[0][5];

    double duration_1;
    if (vkdev->get_pipeline_cache()->clear_shader_cache() != 0)
    {
        fprintf(stderr, "clear shader cache failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }
    if (!test_pipeline_creation(opt, &duration_1))
    {
        fprintf(stderr, "pipeline creation without cache failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }
    fprintf(stdout, "pipeline cache test creation time (without cache): %.2f ms\n", duration_1);
    if (vkdev->get_pipeline_cache()->save_pipeline_cache("vk_pipeline_cache") != 0)
    {
        fprintf(stderr, "save pipeline cache failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }

    ncnn::destroy_gpu_instance();

    ncnn::create_gpu_instance();

    int ret = ncnn::get_gpu_device(0)->get_pipeline_cache()->load_pipeline_cache("vk_pipeline_cache");
    if (ret != 0)
    {
        fprintf(stderr, "load pipeline cache failed\n");
        return false;
    }
    double duration_2;
    if (!test_pipeline_creation(opt, &duration_2))
    {
        fprintf(stderr, "pipeline creation without cache failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }
    fprintf(stdout, "pipeline cache test creation time (with cache): %.2f ms\n", duration_2);
    remove("vk_pipeline_cache");
    ncnn::destroy_gpu_instance();
    return true;
}

bool pipeline_cache_test_corrupted_cache_file()
{
    fprintf(stdout, "Start file corruption test\n");
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);
    // first create and save cache file
    ncnn::create_gpu_instance();
    const int options[][6] = {
        {0, 0, 0, 0, 0, 0},
    };

    ncnn::Option opt{};
    opt.num_threads = 1;
    opt.use_packing_layout = options[0][0];
    opt.use_fp16_packed = options[0][1];
    opt.use_fp16_storage = options[0][2];
    opt.use_fp16_arithmetic = options[0][3];
    opt.use_bf16_storage = options[0][4];
    opt.use_shader_pack8 = options[0][5];

    if (vkdev->get_pipeline_cache()->clear_shader_cache() != 0)
    {
        fprintf(stderr, "clear shader cache failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }
    double duration_1;
    if (!test_pipeline_creation(opt, &duration_1))
    {
        fprintf(stderr, "pipeline creation without cache failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }

    fprintf(stdout, "pipeline cache test creation time (without cache): %.2f ms\n", duration_1);
    if (vkdev->get_pipeline_cache()->save_pipeline_cache("vk_pipeline_cache") != 0)
    {
        fprintf(stderr, "save pipeline cache failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }
    ncnn::destroy_gpu_instance();
    corrupt_file("vk_pipeline_cache");
    ncnn::create_gpu_instance();
    int ret = ncnn::get_gpu_device(0)->get_pipeline_cache()->load_pipeline_cache("vk_pipeline_cache");
    if (ret)
    {
        fprintf(stderr, "load cache after file corruption failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }
    double duration_2;
    if (!test_pipeline_creation(opt, &duration_2))
    {
        fprintf(stderr, "pipeline creation after cache corruption failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }
    fprintf(stdout, "pipeline cache test creation time (after cache corruption): %.2f ms\n", duration_2);
    remove("vk_pipeline_cache");
    ncnn::destroy_gpu_instance();
    return true;
}

bool pipeline_cache_test_multithread_creation()
{
    fprintf(stdout, "Start multi-thread test\n");

    ncnn::create_gpu_instance();
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);

    ncnn::Option opt{};
    opt.num_threads = 1;
    opt.use_packing_layout = 0;
    opt.use_fp16_packed = 1;
    opt.use_fp16_storage = 0;
    opt.use_fp16_arithmetic = 0;
    opt.use_bf16_storage = 1;
    opt.use_shader_pack8 = 0;

    if (vkdev->get_pipeline_cache()->clear_shader_cache() != 0)
    {
        fprintf(stderr, "clear shader cache failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }
    double duration;
    if (!test_pipeline_creation(opt, &duration))
    {
        fprintf(stderr, "pipeline creation failed before multi-thread test\n");
        ncnn::destroy_gpu_instance();
        return false;
    }
    if (vkdev->get_pipeline_cache()->save_pipeline_cache("vk_pipeline_cache") != 0)
    {
        fprintf(stderr, "save pipeline cache failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }
    ncnn::destroy_gpu_instance();

    ncnn::create_gpu_instance();
    vkdev = ncnn::get_gpu_device(0);
    if (vkdev->get_pipeline_cache()->load_pipeline_cache("vk_pipeline_cache") != 0)
    {
        fprintf(stderr, "load pipeline cache failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }

    const int thread_count = 8;
    std::vector<std::future<bool> > futures;
    for (int i = 0; i < thread_count; i++)
    {
        futures.emplace_back(std::async(std::launch::async, [&opt]() {
            return test_pipeline_creation(opt, nullptr);
        }));
    }

    bool all_ok = true;
    for (auto& fut : futures)
    {
        if (!fut.get())
            all_ok = false;
    }

    remove("vk_pipeline_cache");
    ncnn::destroy_gpu_instance();

    if (!all_ok)
    {
        fprintf(stderr, "multi-thread pipeline creation failed\n");
        return false;
    }

    fprintf(stdout, "multi-thread pipeline creation passed\n");
    return true;
}

bool pipeline_cache_test_multithread_save()
{
    fprintf(stdout, "Start multi-thread save test\n");

    ncnn::create_gpu_instance();
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);

    ncnn::Option opt{};
    opt.num_threads = 1;
    opt.use_packing_layout = 0;
    opt.use_fp16_packed = 0;
    opt.use_fp16_storage = 0;
    opt.use_fp16_arithmetic = 0;
    opt.use_bf16_storage = 0;
    opt.use_shader_pack8 = 0;

    if (vkdev->get_pipeline_cache()->clear_shader_cache() != 0)
    {
        fprintf(stderr, "clear shader cache failed\n");
        ncnn::destroy_gpu_instance();
        return false;
    }

    if (!test_pipeline_creation(opt, nullptr))
    {
        fprintf(stderr, "pipeline creation failed before multi-thread save test\n");
        ncnn::destroy_gpu_instance();
        return false;
    }

    const int thread_count = 8;
    std::vector<std::future<int> > futures;
    for (int i = 0; i < thread_count; i++)
    {
        futures.emplace_back(std::async(std::launch::async, [vkdev]() {
            return vkdev->get_pipeline_cache()->save_pipeline_cache("vk_pipeline_cache");
        }));
    }

    bool all_ok = true;
    for (auto& fut : futures)
    {
        if (fut.get() != 0)
            all_ok = false;
    }

    ncnn::destroy_gpu_instance();

    if (!all_ok)
    {
        fprintf(stderr, "multi-thread save_pipeline_cache had errors\n");
        return false;
    }

    ncnn::create_gpu_instance();
    vkdev = ncnn::get_gpu_device(0);
    int ret = vkdev->get_pipeline_cache()->load_pipeline_cache("vk_pipeline_cache");
    remove("vk_pipeline_cache");
    ncnn::destroy_gpu_instance();

    if (ret != 0)
    {
        fprintf(stderr, "cache file after multi-thread save is invalid\n");
        return false;
    }

    fprintf(stdout, "multi-thread save_pipeline_cache passed\n");
    return true;
}

int main()
{
    SRAND(7767517);
    if (!pipeline_cache_test_basic_creation())
    {
        fprintf(stderr, "pipeline cache basic test failed\n");
        return -1;
    }
    if (!pipeline_cache_test_corrupted_cache_file())
    {
        fprintf(stderr, "pipeline cache corrupted file test failed\n");
        return -1;
    }
    if (!pipeline_cache_test_multithread_creation())
    {
        fprintf(stderr, "pipeline cache multi-thread creation test failed\n");
        return -1;
    }
    if (!pipeline_cache_test_multithread_save())
    {
        fprintf(stderr, "pipeline cache multi-thread save test failed\n");
        return -1;
    }
    fprintf(stdout, "All pipeline cache tests passed\n");
    return 0;
}