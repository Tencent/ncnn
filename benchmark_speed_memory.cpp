// Benchmark tool for YOLO26 NCNN operators
// Tests speed and memory efficiency

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

void benchmark_gatherelements()
{
    printf("\n=== GatherElements Benchmark ===\n");
    
    // Test 1: 1D large tensor
    Mat input1(10000);
    float* iptr1 = (float*)input1;
    for (int i = 0; i < 10000; i++) iptr1[i] = (float)i;
    
    Mat indices1(10000);
    int* idx1 = (int*)indices1;
    for (int i = 0; i < 10000; i++) idx1[i] = i % 10000;
    
    Layer* op = create_layer("GatherElements");
    ParamDict pd; pd.set(0, 0); op->load_param(pd);
    
    Option opt;
    opt.num_threads = 4;
    
    std::vector<Mat> bottom(2), top(1);
    bottom[0] = input1;
    bottom[1] = indices1;
    
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
    size_t memory = input1.total() * sizeof(float) + indices1.total() * sizeof(int) + top[0].total() * sizeof(float);
    
    printf("1D (10K elements):\n");
    printf("  Avg time: %.3f ms\n", avg_time);
    printf("  Memory: %.2f KB\n", memory / 1024.0);
    printf("  Throughput: %.2f MB/s\n", (memory / 1024.0 / 1024.0) / (avg_time / 1000.0));
    
    delete op;
}

void benchmark_mod()
{
    printf("\n=== Mod Benchmark ===\n");
    
    Mat a(100000);
    float* aptr = (float*)a;
    for (int i = 0; i < 100000; i++) aptr[i] = (float)i;
    
    Mat b(100000);
    float* bptr = (float*)b;
    for (int i = 0; i < 100000; i++) bptr[i] = 17.0f;
    
    Layer* op = create_layer("Mod");
    ParamDict pd; pd.set(0, 0); op->load_param(pd);
    
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
    size_t memory = (a.total() + b.total() + top[0].total()) * sizeof(float);
    
    printf("100K elements:\n");
    printf("  Avg time: %.3f ms\n", avg_time);
    printf("  Memory: %.2f KB\n", memory / 1024.0);
    printf("  Throughput: %.2f MB/s\n", (memory / 1024.0 / 1024.0) / (avg_time / 1000.0));
    
    delete op;
}

void benchmark_tile()
{
    printf("\n=== Tile Benchmark ===\n");
    
    Mat input(100, 100);
    float* iptr = (float*)input;
    for (int i = 0; i < 10000; i++) iptr[i] = (float)i;
    
    Mat repeats(2);
    int* rptr = (int*)repeats;
    rptr[0] = 2;
    rptr[1] = 2;
    
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
    size_t memory = (input.total() + top[0].total()) * sizeof(float);
    
    printf("100x100 -> 200x200:\n");
    printf("  Avg time: %.3f ms\n", avg_time);
    printf("  Memory: %.2f KB\n", memory / 1024.0);
    printf("  Throughput: %.2f MB/s\n", (memory / 1024.0 / 1024.0) / (avg_time / 1000.0));
    
    delete op;
}

void benchmark_expand()
{
    printf("\n=== Expand Benchmark ===\n");
    
    Mat input(1);
    ((float*)input)[0] = 42.0f;
    
    Mat shape(2);
    int* sptr = (int*)shape;
    sptr[0] = 500;
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
    size_t memory = (input.total() + top[0].total()) * sizeof(float);
    
    printf("1 -> 500x500:\n");
    printf("  Avg time: %.3f ms\n", avg_time);
    printf("  Memory: %.2f KB\n", memory / 1024.0);
    printf("  Throughput: %.2f MB/s\n", (memory / 1024.0 / 1024.0) / (avg_time / 1000.0));
    
    delete op;
}

int main()
{
    printf("================================================================================\n");
    printf("YOLO26 NCNN Operators - Speed & Memory Benchmark\n");
    printf("================================================================================\n");
    
    benchmark_gatherelements();
    benchmark_mod();
    benchmark_tile();
    benchmark_expand();
    
    printf("\n================================================================================\n");
    printf("Benchmark complete!\n");
    printf("================================================================================\n");
    
    return 0;
}
