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
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32

#include "c_api.h"
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) > (b)) ? (b) : (a))

#ifdef _WIN32
double get_current_time()
{
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);

    return pc.QuadPart * 1000.0 / freq.QuadPart;
}
#else  // _WIN32
double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif // _WIN32

static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static _Bool g_enable_cooling_down = true;
static size_t emptydr_read(ncnn_datareader_t dr, void* buf, size_t size)
{
    memset(buf, 0, size);
    return size;
}
void benchmark(const char* comment, int w, int h, int c, ncnn_option_t opt)
{
    ncnn_mat_t in = ncnn_mat_create_3d(w, h, c, NULL);
    ncnn_mat_fill_float(in, 0.01f);

    ncnn_net_t net = ncnn_net_create();
    ncnn_net_set_option(net, opt);
    char parampath[256];
    sprintf(parampath, "%s.param", comment);
    ncnn_net_load_param(net, parampath);

    ncnn_datareader_t dr = ncnn_datareader_create();
    {
        dr->read = emptydr_read;
    }
    ncnn_net_load_model_datareader(net, dr);

    if (g_enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
        Sleep(10 * 1000);
#elif defined(__unix__) || defined(__APPLE__)
        sleep(10);
#elif _POSIX_TIMERS
        struct timespec ts;
        ts.tv_sec = 10;
        ts.tv_nsec = 0;
        nanosleep(&ts, &ts);
#else
        // TODO How to handle it ?
#endif
    }

    ncnn_mat_t out;

    // warm up
    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        ncnn_extractor_t ex = ncnn_extractor_create(net);
        ncnn_extractor_input(ex, "data", in);
        ncnn_extractor_extract(ex, "output", &out);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = get_current_time();

        {
            ncnn_extractor_t ex = ncnn_extractor_create(net);
            ncnn_extractor_input(ex, "data", in);
            ncnn_extractor_extract(ex, "output", &out);
        }

        double end = get_current_time();

        double time = end - start;

        time_min = min(time_min, time);
        time_max = max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg);
}

int main(int argc, char** argv)
{
    int loop_count = 4;
    int num_threads = 1;
    int powersave = 0;
    int gpu_device = -1;
    int cooling_down = 1;

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
        gpu_device = atoi(argv[4]);
    }
    if (argc >= 6)
    {
        cooling_down = atoi(argv[5]);
    }

    g_enable_cooling_down = true;

    g_loop_count = loop_count;

    // default option
    ncnn_option_t opt = ncnn_option_create();
    ncnn_option_set_use_vulkan_compute(opt, 1);
    ncnn_option_set_num_threads(opt, num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "cooling_down = %d\n", (int)g_enable_cooling_down);

    char* model[] = {"squeezenet", "227", "squeezenet_int8", "227", "mobilenet", "224", "mobilenet_int8", "224", "mobilenet_v2", "224", "mobilenet_v3", "224",
                     "shufflenet", "224", "shufflenet_v2", "224", "mnasnet", "224", "proxylessnasnet", "224",
                     "efficientnet_b0", "224", "regnety_400m", "224", "blazeface", "128", "googlenet", "224", "googlenet_int8", "224",
                     "resnet18", "224", "resnet18_int8", "224", "alexnet", "227", "vgg16", "224", "vgg16_int8", "224", "resnet50", "224", "resnet50_int8", "224",
                     "squeezenet_ssd", "300", "squeezenet_ssd_int8", "300", "mobilenet_ssd", "300", "mobilenet_ssd_int8", "300",
                     "mobilenet_yolo", "416", "mobilenetv2_yolov3", "352", "yolov4-tiny", "416"};

    int modellen = sizeof(model) / sizeof(model[0]);
    for (int i = 0; i < modellen; i++)
    {
        char* mmodel = model[i];
        int wh = atoi(model[++i]);
        benchmark(mmodel, wh, wh, 3, opt);
    }

    return 0;
}
