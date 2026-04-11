// Benchmark and correctness test for YOLO26 NCNN operators
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "layer/gatherelements.h"
#include "layer/mod.h"
#include "layer/tile.h"
#include "layer/expand.h"
#include "mat.h"
#include "option.h"
#include "benchmark.h"

// Helper to check if two floats are approximately equal
bool approx_equal(float a, float b, float epsilon = 0.001f)
{
    return std::abs(a - b) < epsilon;
}

// Test GatherElements correctness
int test_gatherelements_correctness()
{
    printf("Testing GatherElements correctness...\n");
    
    // Create 3x4 input matrix
    ncnn::Mat input(3, 4);
    float input_data[] = {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };
    memcpy(input, input_data, 12 * sizeof(float));
    
    // Create 2x4 index matrix (gather along axis 0)
    ncnn::Mat indices(2, 4, (size_t)4u);
    int index_data[] = {
        0, 1, 2, 0,
        2, 1, 0, 1
    };
    memcpy(indices, index_data, 8 * sizeof(int));
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::Layer* op = ncnn::create_layer("GatherElements");
    ncnn::ParamDict pd;
    pd.set(0, 0); // axis=0
    op->load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = input;
    bottom_blobs[1] = indices;
    
    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);
    delete op;
    
    if (ret != 0)
    {
        printf("  ✗ Forward failed\n");
        return -1;
    }
    
    // Expected output (gather along axis 0):
    // Row 0: input[0,0], input[1,1], input[2,2], input[0,3] = 1, 6, 11, 4
    // Row 1: input[2,0], input[1,1], input[0,2], input[1,3] = 9, 6, 3, 8
    float expected[] = {1.0f, 6.0f, 11.0f, 4.0f, 9.0f, 6.0f, 3.0f, 8.0f};
    
    const ncnn::Mat& out = top_blobs[0];
    bool correct = true;
    for (int i = 0; i < 8; i++)
    {
        if (!approx_equal(((const float*)out)[i], expected[i]))
        {
            printf("  ✗ Mismatch at index %d: expected %.1f, got %.1f\n", i, expected[i], ((const float*)out)[i]);
            correct = false;
        }
    }
    
    if (correct)
    {
        printf("  ✓ GatherElements CORRECT\n");
        return 0;
    }
    else
    {
        printf("  ✗ GatherElements INCORRECT\n");
        return -1;
    }
}

// Test Mod correctness
int test_mod_correctness()
{
    printf("Testing Mod correctness...\n");
    
    // Create test data
    ncnn::Mat a(10);
    ncnn::Mat b(10);
    float a_data[] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
    float b_data[] = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    memcpy(a, a_data, 10 * sizeof(float));
    memcpy(b, b_data, 10 * sizeof(float));
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::Layer* op = ncnn::create_layer("Mod");
    ncnn::ParamDict pd;
    pd.set(0, 0); // fmod=0 (Python-style)
    op->load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = a;
    bottom_blobs[1] = b;
    
    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);
    delete op;
    
    if (ret != 0)
    {
        printf("  ✗ Forward failed\n");
        return -1;
    }
    
    // Expected: 10%3=1, 11%3=2, 12%3=0, 13%3=1, 14%3=2, 15%3=0, 16%3=1, 17%3=2, 18%3=0, 19%3=1
    float expected[] = {1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f};
    
    const ncnn::Mat& out = top_blobs[0];
    bool correct = true;
    for (int i = 0; i < 10; i++)
    {
        if (!approx_equal(((const float*)out)[i], expected[i]))
        {
            printf("  ✗ Mismatch at index %d: expected %.1f, got %.1f\n", i, expected[i], ((const float*)out)[i]);
            correct = false;
        }
    }
    
    if (correct)
    {
        printf("  ✓ Mod CORRECT\n");
        return 0;
    }
    else
    {
        printf("  ✗ Mod INCORRECT\n");
        return -1;
    }
}

// Test Tile correctness
int test_tile_correctness()
{
    printf("Testing Tile correctness...\n");
    
    // Create 2x1 input
    ncnn::Mat input(2, 1);
    float input_data[] = {1.0f, 2.0f};
    memcpy(input, input_data, 2 * sizeof(float));
    
    // Create repeats [1, 3]
    ncnn::Mat repeats(2, (size_t)4u);
    int repeats_data[] = {1, 3};
    memcpy(repeats, repeats_data, 2 * sizeof(int));
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::Layer* op = ncnn::create_layer("Tile");
    ncnn::ParamDict pd;
    op->load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = input;
    bottom_blobs[1] = repeats;
    
    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);
    delete op;
    
    if (ret != 0)
    {
        printf("  ✗ Forward failed\n");
        return -1;
    }
    
    // Expected: tile [1; 2] by [1, 3] = [1, 1, 1; 2, 2, 2]
    const ncnn::Mat& out = top_blobs[0];
    if (out.w != 2 || out.h != 3)
    {
        printf("  ✗ Wrong output shape: %d x %d\n", out.w, out.h);
        return -1;
    }
    
    const float* outptr = (const float*)out;
    float expected[] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    
    bool correct = true;
    for (int i = 0; i < 6; i++)
    {
        if (!approx_equal(outptr[i], expected[i]))
        {
            printf("  ✗ Mismatch at index %d: expected %.1f, got %.1f\n", i, expected[i], outptr[i]);
            correct = false;
        }
    }
    
    if (correct)
    {
        printf("  ✓ Tile CORRECT\n");
        return 0;
    }
    else
    {
        printf("  ✗ Tile INCORRECT\n");
        return -1;
    }
}

// Test Expand correctness
int test_expand_correctness()
{
    printf("Testing Expand correctness...\n");
    
    // Create 1x1 input
    ncnn::Mat input(1, 1);
    ((float*)input)[0] = 42.0f;
    
    // Create shape [3]
    ncnn::Mat shape(1, (size_t)4u);
    ((int*)shape)[0] = 3;
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::Layer* op = ncnn::create_layer("Expand");
    ncnn::ParamDict pd;
    op->load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = input;
    bottom_blobs[1] = shape;
    
    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);
    delete op;
    
    if (ret != 0)
    {
        printf("  ✗ Forward failed\n");
        return -1;
    }
    
    // Expected: expand [42] to [42, 42, 42]
    const ncnn::Mat& out = top_blobs[0];
    if (out.w != 3 || out.h != 1 || out.c != 1)
    {
        printf("  ✗ Wrong output shape: %d x %d x %d\n", out.w, out.h, out.c);
        return -1;
    }
    
    bool correct = true;
    for (int i = 0; i < 3; i++)
    {
        if (!approx_equal(((const float*)out)[i], 42.0f))
        {
            printf("  ✗ Mismatch at index %d: expected 42.0, got %.1f\n", i, ((const float*)out)[i]);
            correct = false;
        }
    }
    
    if (correct)
    {
        printf("  ✓ Expand CORRECT\n");
        return 0;
    }
    else
    {
        printf("  ✗ Expand INCORRECT\n");
        return -1;
    }
}

// Benchmark GatherElements
int benchmark_gatherelements()
{
    printf("\nBenchmarking GatherElements...\n");
    
    // Large test case
    ncnn::Mat input(100, 200);
    ncnn::Mat indices(50, 200, (size_t)4u);
    
    // Fill with random data
    for (int i = 0; i < (int)input.total(); i++)
        ((float*)input)[i] = (float)i;
    
    for (int i = 0; i < (int)indices.total(); i++)
        ((int*)indices)[i] = i % 100;
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::Layer* op = ncnn::create_layer("GatherElements");
    ncnn::ParamDict pd;
    pd.set(0, 0);
    op->load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = input;
    bottom_blobs[1] = indices;
    
    std::vector<ncnn::Mat> top_blobs(1);
    
    // Warmup
    op->forward(bottom_blobs, top_blobs, opt);
    
    // Benchmark
    double start = ncnn::get_current_time();
    for (int i = 0; i < 100; i++)
    {
        op->forward(bottom_blobs, top_blobs, opt);
    }
    double end = ncnn::get_current_time();
    
    double avg_time = (end - start) / 100.0;
    size_t memory = input.total() * sizeof(float) + indices.total() * sizeof(int) + top_blobs[0].total() * sizeof(float);
    
    printf("  Input: %d x %d, Indices: %d x %d\n", input.w, input.h, indices.w, indices.h);
    printf("  Output: %d x %d\n", top_blobs[0].w, top_blobs[0].h);
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Memory: %.2f KB\n", memory / 1024.0);
    printf("  Throughput: %.2f MB/s\n", (memory / 1024.0 / 1024.0) / (avg_time / 1000.0));
    
    delete op;
    return 0;
}

// Benchmark Mod
int benchmark_mod()
{
    printf("\nBenchmarking Mod...\n");
    
    // Large test case
    ncnn::Mat a(10000);
    ncnn::Mat b(10000);
    
    for (int i = 0; i < 10000; i++)
    {
        ((float*)a)[i] = (float)i;
        ((float*)b)[i] = 17.0f;
    }
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::Layer* op = ncnn::create_layer("Mod");
    ncnn::ParamDict pd;
    pd.set(0, 0);
    op->load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = a;
    bottom_blobs[1] = b;
    
    std::vector<ncnn::Mat> top_blobs(1);
    
    // Warmup
    op->forward(bottom_blobs, top_blobs, opt);
    
    // Benchmark
    double start = ncnn::get_current_time();
    for (int i = 0; i < 100; i++)
    {
        op->forward(bottom_blobs, top_blobs, opt);
    }
    double end = ncnn::get_current_time();
    
    double avg_time = (end - start) / 100.0;
    size_t memory = (a.total() + b.total() + top_blobs[0].total()) * sizeof(float);
    
    printf("  Size: %d elements\n", 10000);
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Memory: %.2f KB\n", memory / 1024.0);
    printf("  Throughput: %.2f MB/s\n", (memory / 1024.0 / 1024.0) / (avg_time / 1000.0));
    
    delete op;
    return 0;
}

// Benchmark Tile
int benchmark_tile()
{
    printf("\nBenchmarking Tile...\n");
    
    // Test case
    ncnn::Mat input(50, 100);
    ncnn::Mat repeats(2, (size_t)4u);
    ((int*)repeats)[0] = 2;
    ((int*)repeats)[1] = 3;
    
    for (int i = 0; i < (int)input.total(); i++)
        ((float*)input)[i] = (float)i;
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::Layer* op = ncnn::create_layer("Tile");
    ncnn::ParamDict pd;
    op->load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = input;
    bottom_blobs[1] = repeats;
    
    std::vector<ncnn::Mat> top_blobs(1);
    
    // Warmup
    op->forward(bottom_blobs, top_blobs, opt);
    
    // Benchmark
    double start = ncnn::get_current_time();
    for (int i = 0; i < 100; i++)
    {
        op->forward(bottom_blobs, top_blobs, opt);
    }
    double end = ncnn::get_current_time();
    
    double avg_time = (end - start) / 100.0;
    size_t memory = input.total() * sizeof(float) + top_blobs[0].total() * sizeof(float);
    
    printf("  Input: %d x %d, Repeats: [2, 3]\n", input.w, input.h);
    printf("  Output: %d x %d\n", top_blobs[0].w, top_blobs[0].h);
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Memory: %.2f KB\n", memory / 1024.0);
    printf("  Throughput: %.2f MB/s\n", (memory / 1024.0 / 1024.0) / (avg_time / 1000.0));
    
    delete op;
    return 0;
}

// Benchmark Expand
int benchmark_expand()
{
    printf("\nBenchmarking Expand...\n");
    
    // Test case
    ncnn::Mat input(50, 100);
    ncnn::Mat shape(2, (size_t)4u);
    ((int*)shape)[0] = 50;
    ((int*)shape)[1] = 100;
    
    for (int i = 0; i < (int)input.total(); i++)
        ((float*)input)[i] = (float)i;
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::Layer* op = ncnn::create_layer("Expand");
    ncnn::ParamDict pd;
    op->load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = input;
    bottom_blobs[1] = shape;
    
    std::vector<ncnn::Mat> top_blobs(1);
    
    // Warmup
    op->forward(bottom_blobs, top_blobs, opt);
    
    // Benchmark
    double start = ncnn::get_current_time();
    for (int i = 0; i < 100; i++)
    {
        op->forward(bottom_blobs, top_blobs, opt);
    }
    double end = ncnn::get_current_time();
    
    double avg_time = (end - start) / 100.0;
    size_t memory = input.total() * sizeof(float) + top_blobs[0].total() * sizeof(float);
    
    printf("  Input: %d x %d, Shape: [50, 100]\n", input.w, input.h);
    printf("  Output: %d x %d\n", top_blobs[0].w, top_blobs[0].h);
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Memory: %.2f KB\n", memory / 1024.0);
    printf("  Throughput: %.2f MB/s\n", (memory / 1024.0 / 1024.0) / (avg_time / 1000.0));
    
    delete op;
    return 0;
}

int main()
{
    printf("================================================================================\n");
    printf("YOLO26 NCNN Operators - Correctness & Benchmark Test\n");
    printf("================================================================================\n\n");
    
    // Correctness tests
    printf("CORRECTNESS TESTS\n");
    printf("--------------------------------------------------------------------------------\n");
    
    int passed = 0;
    int total = 0;
    
    total++; if (test_gatherelements_correctness() == 0) passed++;
    total++; if (test_mod_correctness() == 0) passed++;
    total++; if (test_tile_correctness() == 0) passed++;
    total++; if (test_expand_correctness() == 0) passed++;
    
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("Correctness: %d/%d tests passed\n", passed, total);
    printf("--------------------------------------------------------------------------------\n\n");
    
    if (passed != total)
    {
        printf("❌ Some correctness tests FAILED - stopping benchmarks\n");
        return 1;
    }
    
    // Benchmarks
    printf("BENCHMARKS\n");
    printf("--------------------------------------------------------------------------------\n");
    
    benchmark_gatherelements();
    benchmark_mod();
    benchmark_tile();
    benchmark_expand();
    
    printf("\n");
    printf("================================================================================\n");
    printf("✅ All correctness tests PASSED!\n");
    printf("================================================================================\n");
    
    return 0;
}
