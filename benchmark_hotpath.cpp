// Aggressive benchmark for YOLO26 NCNN operators - Hot Path Optimization
// Tests maximum throughput with various input sizes

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include "layer/gatherelements.h"
#include "layer/mod.h"
#include "layer/tile.h"
#include "layer/expand.h"
#include "mat.h"
#include "option.h"
#include "benchmark.h"

using namespace ncnn;

void benchmark_gatherelements_hotpath()
{
    printf("\n=== GatherElements HOT PATH Benchmark ===\n");
    
    // Test 1: 1D large tensor (hot path)
    printf("\n1D Hot Path:\n");
    for (int size = 10000; size <= 100000; size += 30000)
    {
        Mat input(size);
        float* iptr = (float*)input;
        for (int i = 0; i < size; i++) iptr[i] = (float)i;
        
        Mat indices(size);
        int* idx = (int*)indices;
        for (int i = 0; i < size; i++) idx[i] = i % size;
        
        Layer* op = create_layer("GatherElements");
        ParamDict pd; pd.set(0, 0); op->load_param(pd);
        
        Option opt;
        opt.num_threads = 4;
        
        std::vector<Mat> bottom(2), top(1);
        bottom[0] = input;
        bottom[1] = indices;
        
        // Warmup
        op->forward(bottom, top, opt);
        
        // Benchmark
        double start = get_current_time();
        for (int i = 0; i < 100; i++)
        {
            op->forward(bottom, top, opt);
        }
        double end = get_current_time();
        
        double avg_time = (end - start) / 100.0;
        size_t memory = (input.total() * sizeof(float) + indices.total() * sizeof(int) + top[0].total() * sizeof(float)) / 1024.0;
        
        printf("  %6d elements: %6.3f ms, %6.2f KB, %7.2f MB/s\n", 
               size, avg_time, memory, (memory / 1024.0) / (avg_time / 1000.0));
        
        delete op;
    }
}

void benchmark_mod_hotpath()
{
    printf("\n=== Mod HOT PATH Benchmark ===\n");
    
    printf("\nC-style Fmod (Optimized):\n");
    for (int size = 10000; size <= 100000; size += 30000)
    {
        Mat a(size);
        float* aptr = (float*)a;
        for (int i = 0; i < size; i++) aptr[i] = (float)i;
        
        Mat b(size);
        float* bptr = (float*)b;
        for (int i = 0; i < size; i++) bptr[i] = 17.0f;
        
        Layer* op = create_layer("Mod");
        ParamDict pd; pd.set(0, 1); op->load_param(pd);
        
        Option opt;
        opt.num_threads = 4;
        
        std::vector<Mat> bottom(2), top(1);
        bottom[0] = a;
        bottom[1] = b;
        
        // Warmup
        op->forward(bottom, top, opt);
        
        // Benchmark
        double start = get_current_time();
        for (int i = 0; i < 100; i++)
        {
            op->forward(bottom, top, opt);
        }
        double end = get_current_time();
        
        double avg_time = (end - start) / 100.0;
        size_t memory = ((a.total() + b.total() + top[0].total()) * sizeof(float)) / 1024.0;
        
        printf("  %6d elements: %6.3f ms, %6.2f KB, %7.2f MB/s\n", 
               size, avg_time, memory, (memory / 1024.0) / (avg_time / 1000.0));
        
        delete op;
    }
}

void benchmark_tile_hotpath()
{
    printf("\n=== Tile HOT PATH Benchmark ===\n");
    
    printf("\nHorizontal Tiling (repeat_w > 1):\n");
    for (int w = 100; w <= 500; w += 200)
    {
        Mat input(w, 100);
        float* iptr = (float*)input;
        for (int i = 0; i < w * 100; i++) iptr[i] = (float)i;
        
        Mat repeats(2);
        int* rptr = (int*)repeats;
        rptr[0] = 4;  // repeat_w = 4
        rptr[1] = 1;  // repeat_h = 1
        
        Layer* op = create_layer("Tile");
        op->load_param(ParamDict());
        
        Option opt;
        opt.num_threads = 4;
        
        std::vector<Mat> bottom(2), top(1);
        bottom[0] = input;
        bottom[1] = repeats;
        
        // Warmup
        op->forward(bottom, top, opt);
        
        // Benchmark
        double start = get_current_time();
        for (int i = 0; i < 100; i++)
        {
            op->forward(bottom, top, opt);
        }
        double end = get_current_time();
        
        double avg_time = (end - start) / 100.0;
        size_t memory = ((input.total() + top[0].total()) * sizeof(float)) / 1024.0;
        
        printf("  %3dx100 -> %3dx100: %6.3f ms, %6.2f KB, %7.2f MB/s\n", 
               w, w * 4, avg_time, memory, (memory / 1024.0) / (avg_time / 1000.0));
        
        delete op;
    }
    
    printf("\nVertical Tiling (repeat_h > 1):\n");
    for (int h = 100; h <= 500; h += 200)
    {
        Mat input(100, h);
        float* iptr = (float*)input;
        for (int i = 0; i < 100 * h; i++) iptr[i] = (float)i;
        
        Mat repeats(2);
        int* rptr = (int*)repeats;
        rptr[0] = 1;  // repeat_w = 1
        rptr[1] = 4;  // repeat_h = 4
        
        Layer* op = create_layer("Tile");
        op->load_param(ParamDict());
        
        Option opt;
        opt.num_threads = 4;
        
        std::vector<Mat> bottom(2), top(1);
        bottom[0] = input;
        bottom[1] = repeats;
        
        // Warmup
        op->forward(bottom, top, opt);
        
        // Benchmark
        double start = get_current_time();
        for (int i = 0; i < 100; i++)
        {
            op->forward(bottom, top, opt);
        }
        double end = get_current_time();
        
        double avg_time = (end - start) / 100.0;
        size_t memory = ((input.total() + top[0].total()) * sizeof(float)) / 1024.0;
        
        printf("  100x%3d -> 100x%3d: %6.3f ms, %6.2f KB, %7.2f MB/s\n", 
               h, h * 4, avg_time, memory, (memory / 1024.0) / (avg_time / 1000.0));
        
        delete op;
    }
}

void benchmark_expand_hotpath()
{
    printf("\n=== Expand HOT PATH Benchmark ===\n");
    
    printf("\nSingle Value Broadcast:\n");
    for (int size = 10000; size <= 100000; size += 30000)
    {
        Mat input(1);
        ((float*)input)[0] = 42.0f;
        
        Mat shape(1);
        ((int*)shape)[0] = size;
        
        Layer* op = create_layer("Expand");
        op->load_param(ParamDict());
        
        Option opt;
        opt.num_threads = 4;
        
        std::vector<Mat> bottom(2), top(1);
        bottom[0] = input;
        bottom[1] = shape;
        
        // Warmup
        op->forward(bottom, top, opt);
        
        // Benchmark
        double start = get_current_time();
        for (int i = 0; i < 100; i++)
        {
            op->forward(bottom, top, opt);
        }
        double end = get_current_time();
        
        double avg_time = (end - start) / 100.0;
        size_t memory = ((input.total() + top[0].total()) * sizeof(float)) / 1024.0;
        
        printf("  1 -> %6d: %6.3f ms, %6.2f KB, %7.2f MB/s\n", 
               size, avg_time, memory, (memory / 1024.0) / (avg_time / 1000.0));
        
        delete op;
    }
    
    printf("\nRow Vector to Matrix:\n");
    for (int w = 100; w <= 500; w += 200)
    {
        Mat input(w, 1);
        float* iptr = (float*)input;
        for (int i = 0; i < w; i++) iptr[i] = (float)i;
        
        Mat shape(2);
        int* sptr = (int*)shape;
        sptr[0] = w;
        sptr[1] = 500;
        
        Layer* op = create_layer("Expand");
        op->load_param(ParamDict());
        
        Option opt;
        opt.num_threads = 4;
        
        std::vector<Mat> bottom(2), top(1);
        bottom[0] = input;
        bottom[1] = shape;
        
        // Warmup
        op->forward(bottom, top, opt);
        
        // Benchmark
        double start = get_current_time();
        for (int i = 0; i < 100; i++)
        {
            op->forward(bottom, top, opt);
        }
        double end = get_current_time();
        
        double avg_time = (end - start) / 100.0;
        size_t memory = ((input.total() + top[0].total()) * sizeof(float)) / 1024.0;
        
        printf("  %3d -> %3dx500: %6.3f ms, %6.2f KB, %7.2f MB/s\n", 
               w, w, avg_time, memory, (memory / 1024.0) / (avg_time / 1000.0));
        
        delete op;
    }
}

int main()
{
    printf("================================================================================\n");
    printf("YOLO26 NCNN - AGGRESSIVE HOT PATH OPTIMIZATION BENCHMARK\n");
    printf("================================================================================\n");
    
    benchmark_gatherelements_hotpath();
    benchmark_mod_hotpath();
    benchmark_tile_hotpath();
    benchmark_expand_hotpath();
    
    printf("\n================================================================================\n");
    printf("Benchmark complete!\n");
    printf("================================================================================\n");
    
    return 0;
}
