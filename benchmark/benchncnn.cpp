// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <float.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"

#ifndef NCNN_SIMPLESTL
#include <vector>
#endif

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

class Benchmark
{
public:
    Benchmark();

    ~Benchmark();

    void add_opt(int num_threads, bool use_vulkan_compute);

    void run(const char* comment, const std::vector<ncnn::Mat>& _in, bool fixed_path = true);
    void run(const char* comment, const ncnn::Mat& _in, bool fixed_path = true);

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
#if NCNN_VULKAN
    ncnn::VulkanDevice* vkdev;
    ncnn::VkAllocator* blob_vkallocator;
    ncnn::VkAllocator* staging_vkallocator;
#endif // NCNN_VULKAN
    int warmup_loop_count;
    int loop_count;
    bool enable_cooling_down;

private:
    struct TimeStamps
    {
        double real_ms;
        double user_ms;
        double sys_ms;
    };

    static TimeStamps get_time_stamps();

    void run(const char* comment, const std::vector<ncnn::Mat>& _in, const ncnn::Option& opt, bool fixed_path);

    // We can have multiple of Option, the first one is for baseline or reference, the other Options are candidate.
    // For example:
    // 0. C++ without assembly optimization
    // 1. Enable SSE3 but not AVX2
    // 2. Enable AVX2
    // 3. Enable Vulkan
    // and so on.
    std::vector<ncnn::Option> opts;

    const char* prev_comment;
    double prev_time_avg;
    double prev_user_avg;
    double prev_sys_avg;
};

Benchmark::Benchmark()
    :
#if NCNN_VULKAN
    vkdev(NULL),
    blob_vkallocator(NULL),
    staging_vkallocator(NULL),
#endif
    warmup_loop_count(8),
    loop_count(4),
    enable_cooling_down(true),
    prev_comment(NULL),
    prev_time_avg(0),
    prev_user_avg(0),
    prev_sys_avg(0)
{
}

Benchmark::~Benchmark()
{
#if NCNN_VULKAN
    delete blob_vkallocator;
    delete staging_vkallocator;
#endif // NCNN_VULKAN
}

void Benchmark::add_opt(int num_threads, bool use_vulkan_compute)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &blob_pool_allocator;
    opt.workspace_allocator = &workspace_pool_allocator;
#if NCNN_VULKAN
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;
#endif // NCNN_VULKAN
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_vulkan_compute = use_vulkan_compute;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;

    opts.push_back(opt);
}

void Benchmark::run(const char* comment, const std::vector<ncnn::Mat>& _in, const ncnn::Option& opt, bool fixed_path)
{
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        blob_vkallocator->clear();
        staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    ncnn::Net net;

    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(vkdev);
    }
#endif // NCNN_VULKAN

#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working/"
#else
#define MODEL_DIR ""
#endif

    if (fixed_path)
    {
        char parampath[256];
        sprintf(parampath, MODEL_DIR "%s.param", comment);
        net.load_param(parampath);
    }
    else
    {
        net.load_param(comment);
    }

    DataReaderFromEmpty dr;
    net.load_model(dr);

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    if (enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
        ncnn::sleep(10 * 1000);
    }

    if (input_names.size() > _in.size())
    {
        fprintf(stderr, "input %ld tensors while model has %ld inputs\n", _in.size(), input_names.size());
        return;
    }

    // initialize input
    for (size_t j = 0; j < input_names.size(); ++j)
    {
        ncnn::Mat in = _in[j];
        in.fill(0.01f);
    }

    // warm up
    for (int i = 0; i < warmup_loop_count; i++)
    {
        ncnn::Extractor ex = net.create_extractor();
        for (size_t j = 0; j < input_names.size(); ++j)
        {
            ncnn::Mat in = _in[j];
            ex.input(input_names[j], in);
        }

        for (size_t j = 0; j < output_names.size(); ++j)
        {
            ncnn::Mat out;
            ex.extract(output_names[j], out);
        }
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;
    double user_time_avg = 0;
    double sys_time_avg = 0;

    for (int i = 0; i < loop_count; i++)
    {
        TimeStamps t1 = get_time_stamps();
        {
            ncnn::Extractor ex = net.create_extractor();
            for (size_t j = 0; j < input_names.size(); ++j)
            {
                ncnn::Mat in = _in[j];
                ex.input(input_names[j], in);
            }

            for (size_t j = 0; j < output_names.size(); ++j)
            {
                ncnn::Mat out;
                ex.extract(output_names[j], out);
            }
        }

        TimeStamps t2 = get_time_stamps();
        double time = t2.real_ms - t1.real_ms;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
        user_time_avg += t2.user_ms - t1.user_ms;
        sys_time_avg += t2.sys_ms - t1.sys_ms;
    }

    time_avg /= loop_count;
    user_time_avg /= loop_count;
    sys_time_avg /= loop_count;

    if (opts.size() == 1)
    {
        // Keep the old format
        fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg);
        return;
    }

    fprintf(stderr, "%20s %s min = %7.2f  max = %7.2f  avg = %7.2f  user = %7.2f  sys = %7.2f",
            comment, opt.use_vulkan_compute ? "gpu" : "cpu",
            time_min, time_max, time_avg,
            user_time_avg, sys_time_avg);

    if (prev_comment != NULL && strcmp(prev_comment, comment) == 0)
    {
        // Relative speed compare to baseline
        double ratio = prev_time_avg / time_avg;
        fprintf(stderr, "  speed_ratio = %6.2fx", ratio);
        if (prev_user_avg + prev_sys_avg > 0)
        {
            ratio = 100.0 * (user_time_avg + sys_time_avg) / (prev_user_avg + prev_sys_avg);
            fprintf(stderr, "  cpu_ratio = %6.2f%%", ratio);
        }
    }
    else
    {
        prev_comment = comment;
        prev_time_avg = time_avg;
        prev_user_avg = user_time_avg;
        prev_sys_avg = sys_time_avg;
    }
    fprintf(stderr, "\n");
}

void Benchmark::run(const char* comment, const ncnn::Mat& _in, bool fixed_path)
{
    std::vector<ncnn::Mat> inputs;
    inputs.push_back(_in);
    run(comment, inputs, fixed_path);
}

void Benchmark::run(const char* comment, const std::vector<ncnn::Mat>& _in, bool fixed_path)
{
    for (size_t i = 0; i < opts.size(); i++)
    {
        run(comment, _in, opts[i], fixed_path);
    }
}

Benchmark::TimeStamps Benchmark::get_time_stamps()
{
    TimeStamps time_stamps = {
        ncnn::get_current_time(),
    };
#if defined(__linux__) || defined(__APPLE__)
    rusage usage = {0};
    getrusage(RUSAGE_SELF, &usage);
    time_stamps.user_ms = (usage.ru_utime.tv_sec * 1000.0) + usage.ru_utime.tv_usec / 1000.0;
    time_stamps.sys_ms = (usage.ru_stime.tv_sec * 1000.0) + usage.ru_stime.tv_usec / 1000.0;
#elif defined(_WIN32)
    HANDLE proc;
    FILETIME c, e, k, u;
    proc = GetCurrentProcess();
    GetProcessTimes(proc, &c, &e, &k, &u);
    time_stamps.user_ms = ((int64_t)u.dwHighDateTime << 32 | u.dwLowDateTime) / 10000.0;
    time_stamps.sys_ms = ((int64_t)k.dwHighDateTime << 32 | k.dwLowDateTime) / 10000.0;
#else
    time_stamps.user_ms = time_stamps.sys_ms = 0;
#endif
    return time_stamps;
}

void show_usage()
{
    fprintf(stderr, "Usage: benchncnn [loop count] [num threads] [powersave] [gpu device] [cooling down] [(key=value)...]\n");
    fprintf(stderr, "  param=model.param\n");
    fprintf(stderr, "  shape=[227,227,3],...\n");
}

static std::vector<ncnn::Mat> parse_shape_list(char* s)
{
    std::vector<std::vector<int> > shapes;
    std::vector<ncnn::Mat> mats;

    char* pch = strtok(s, "[]");
    while (pch != NULL)
    {
        // parse a,b,c
        int v;
        int nconsumed = 0;
        int nscan = sscanf(pch, "%d%n", &v, &nconsumed);
        if (nscan == 1)
        {
            // ok we get shape
            pch += nconsumed;

            std::vector<int> s;
            s.push_back(v);

            nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            while (nscan == 1)
            {
                pch += nconsumed;

                s.push_back(v);

                nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            }

            // shape end
            shapes.push_back(s);
        }

        pch = strtok(NULL, "[]");
    }

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        const std::vector<int>& shape = shapes[i];
        switch (shape.size())
        {
        case 4:
            mats.push_back(ncnn::Mat(shape[0], shape[1], shape[2], shape[3]));
            break;
        case 3:
            mats.push_back(ncnn::Mat(shape[0], shape[1], shape[2]));
            break;
        case 2:
            mats.push_back(ncnn::Mat(shape[0], shape[1]));
            break;
        case 1:
            mats.push_back(ncnn::Mat(shape[0]));
            break;
        default:
            fprintf(stderr, "unsupported input shape size %ld\n", shape.size());
            break;
        }
    }
    return mats;
}

int main(int argc, char** argv)
{
    int loop_count = 4;
    int num_threads = ncnn::get_physical_big_cpu_count();
    int powersave = 2;
    int gpu_device = -1;
    bool cpu_and_gpu = false;
    int cooling_down = 1;
    char* model = 0;
    std::vector<ncnn::Mat> inputs;

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-' && argv[i][1] == 'h')
        {
            show_usage();
            return -1;
        }

        if (strcmp(argv[i], "--help") == 0)
        {
            show_usage();
            return -1;
        }
    }

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        // -1,0     benchmark with CPU and GPU both:
        // 0        benchmark with GPU
        // -1       benchmark with CPU only
        int n[2] = {-1, -1};
        if (sscanf(argv[4], "%d,%d", &n[0], &n[1]) == 2)
        {
            cpu_and_gpu = true;
            gpu_device = n[1];
        }
        else
        {
            gpu_device = n[0];
        }
    }
    if (argc >= 6)
    {
        cooling_down = atoi(argv[5]);
    }

    for (int i = 6; i < argc; i++)
    {
        // key=value
        char* kv = argv[i];

        char* eqs = strchr(kv, '=');
        if (eqs == NULL)
        {
            fprintf(stderr, "unrecognized arg %s\n", kv);
            continue;
        }

        // split k v
        eqs[0] = '\0';
        const char* key = kv;
        char* value = eqs + 1;

        if (strcmp(key, "param") == 0)
            model = value;
        if (strcmp(key, "shape") == 0)
            inputs = parse_shape_list(value);
    }

    if (model && inputs.empty())
    {
        fprintf(stderr, "input tensor shape empty!\n");
        return -1;
    }

#ifdef __EMSCRIPTEN__
    EM_ASM(
        FS.mkdir('/working');
        FS.mount(NODEFS, {root: '.'}, '/working'););
#endif // __EMSCRIPTEN__

    bool use_vulkan_compute = gpu_device != -1;

    Benchmark bench;
    bench.enable_cooling_down = cooling_down != 0;

    bench.loop_count = loop_count;

    bench.blob_pool_allocator.set_size_compare_ratio(0.f);
    bench.workspace_pool_allocator.set_size_compare_ratio(0.f);

#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        bench.warmup_loop_count = 10;

        bench.vkdev = ncnn::get_gpu_device(gpu_device);

        bench.blob_vkallocator = new ncnn::VkBlobAllocator(bench.vkdev);
        bench.staging_vkallocator = new ncnn::VkStagingAllocator(bench.vkdev);
    }
#endif // NCNN_VULKAN

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", bench.loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int)bench.enable_cooling_down);

    if (cpu_and_gpu)
    {
        bench.add_opt(num_threads, false);
        bench.add_opt(num_threads, true);
    }
    else
    {
        bench.add_opt(num_threads, use_vulkan_compute);
    }

    if (model != 0)
    {
        // run user defined benchmark
        bench.run(model, inputs, false);
    }
    else
    {
        // run default cases
        bench.run("squeezenet", ncnn::Mat(227, 227, 3));

        bench.run("squeezenet_int8", ncnn::Mat(227, 227, 3));

        bench.run("mobilenet", ncnn::Mat(224, 224, 3));

        bench.run("mobilenet_int8", ncnn::Mat(224, 224, 3));

        bench.run("mobilenet_v2", ncnn::Mat(224, 224, 3));

        // benchmark("mobilenet_v2_int8", ncnn::Mat(224, 224, 3));

        bench.run("mobilenet_v3", ncnn::Mat(224, 224, 3));

        bench.run("shufflenet", ncnn::Mat(224, 224, 3));

        bench.run("shufflenet_v2", ncnn::Mat(224, 224, 3));

        bench.run("mnasnet", ncnn::Mat(224, 224, 3));

        bench.run("proxylessnasnet", ncnn::Mat(224, 224, 3));

        bench.run("efficientnet_b0", ncnn::Mat(224, 224, 3));

        bench.run("efficientnetv2_b0", ncnn::Mat(224, 224, 3));

        bench.run("regnety_400m", ncnn::Mat(224, 224, 3));

        bench.run("blazeface", ncnn::Mat(128, 128, 3));

        bench.run("googlenet", ncnn::Mat(224, 224, 3));

        bench.run("googlenet_int8", ncnn::Mat(224, 224, 3));

        bench.run("resnet18", ncnn::Mat(224, 224, 3));

        bench.run("resnet18_int8", ncnn::Mat(224, 224, 3));

        bench.run("alexnet", ncnn::Mat(227, 227, 3));

        bench.run("vgg16", ncnn::Mat(224, 224, 3));

        bench.run("vgg16_int8", ncnn::Mat(224, 224, 3));

        bench.run("resnet50", ncnn::Mat(224, 224, 3));

        bench.run("resnet50_int8", ncnn::Mat(224, 224, 3));

        bench.run("squeezenet_ssd", ncnn::Mat(300, 300, 3));

        bench.run("squeezenet_ssd_int8", ncnn::Mat(300, 300, 3));

        bench.run("mobilenet_ssd", ncnn::Mat(300, 300, 3));

        bench.run("mobilenet_ssd_int8", ncnn::Mat(300, 300, 3));

        bench.run("mobilenet_yolo", ncnn::Mat(416, 416, 3));

        bench.run("mobilenetv2_yolov3", ncnn::Mat(352, 352, 3));

        bench.run("yolov4-tiny", ncnn::Mat(416, 416, 3));

        bench.run("nanodet_m", ncnn::Mat(320, 320, 3));

        bench.run("yolo-fastest-1.1", ncnn::Mat(320, 320, 3));

        bench.run("yolo-fastestv2", ncnn::Mat(352, 352, 3));

        bench.run("vision_transformer", ncnn::Mat(384, 384, 3));

        bench.run("FastestDet", ncnn::Mat(352, 352, 3));
    }

    return 0;
}
