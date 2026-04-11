// Comprehensive test suite for YOLO26 NCNN operators
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "layer/gatherelements.h"
#include "layer/mod.h"
#include "layer/tile.h"
#include "layer/expand.h"
#include "mat.h"
#include "option.h"

bool approx_equal(float a, float b, float epsilon = 0.001f)
{
    return std::abs(a - b) < epsilon;
}

ncnn::Mat create_int_mat(int w, int h, int c, const int* data)
{
    ncnn::Mat mat(w, h, c, (size_t)4u);
    int* ptr = (int*)mat;
    int total = w * h * c;
    for (int i = 0; i < total; i++)
        ptr[i] = data[i];
    return mat;
}

ncnn::Mat create_float_mat(int w, int h, int c, const float* data)
{
    ncnn::Mat mat(w, h, c);
    float* ptr = (float*)mat;
    int total = w * h * c;
    for (int i = 0; i < total; i++)
        ptr[i] = data[i];
    return mat;
}

// GATHERELEMENTS - ncnn uses w x h layout, axis=0 means width dimension
int test_gatherelements_basic()
{
    printf("Testing GatherElements basic (axis=0)...\n");
    
    // Input: w=3, h=4
    float input_data[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
    ncnn::Mat input = create_float_mat(3, 4, 1, input_data);
    
    // Indices: w=2, h=2
    int index_data[] = {0,1, 2,0};
    ncnn::Mat indices = create_int_mat(2, 2, 1, index_data);
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::Layer* op = ncnn::create_layer("GatherElements");
    ncnn::ParamDict pd;
    pd.set(0, 0); // axis=0 (width)
    op->load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = input;
    bottom_blobs[1] = indices;
    
    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);
    delete op;
    
    if (ret != 0) { printf("  ✗ Forward failed\n"); return -1; }
    
    // Expected: output[x,y] = input[indices[x,y], y]
    // [0,0]=input[0,0]=1, [1,0]=input[1,0]=2
    // [0,1]=input[2,1]=6, [1,1]=input[0,1]=4
    float expected[] = {1.0f, 2.0f, 6.0f, 4.0f};
    const ncnn::Mat& out = top_blobs[0];
    const float* out_ptr = (const float*)out;
    
    bool correct = true;
    for (int i = 0; i < 4; i++)
    {
        if (!approx_equal(out_ptr[i], expected[i]))
        {
            printf("  ✗ Mismatch at %d: exp %.1f, got %.1f\n", i, expected[i], out_ptr[i]);
            correct = false;
        }
    }
    
    printf(correct ? "  ✓ PASS\n" : "  ✗ FAIL\n");
    return correct ? 0 : -1;
}

int test_gatherelements_axis1()
{
    printf("Testing GatherElements (axis=1)...\n");
    
    // Input: w=2, h=3
    float input_data[] = {1,2, 3,4, 5,6};
    ncnn::Mat input = create_float_mat(2, 3, 1, input_data);
    
    // Indices: w=2, h=2
    int index_data[] = {0,1, 1,0};
    ncnn::Mat indices = create_int_mat(2, 2, 1, index_data);
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::Layer* op = ncnn::create_layer("GatherElements");
    ncnn::ParamDict pd;
    pd.set(0, 1); // axis=1 (height)
    op->load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = input;
    bottom_blobs[1] = indices;
    
    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);
    delete op;
    
    if (ret != 0) { printf("  ✗ Forward failed\n"); return -1; }
    
    // Expected: output[x,y] = input[x, indices[x,y]]
    // [0,0]=input[0,0]=1, [1,0]=input[1,1]=4
    // [0,1]=input[0,1]=3, [1,1]=input[1,0]=2
    float expected[] = {1.0f, 4.0f, 3.0f, 2.0f};
    const ncnn::Mat& out = top_blobs[0];
    const float* out_ptr = (const float*)out;
    
    bool correct = true;
    for (int i = 0; i < 4; i++)
    {
        if (!approx_equal(out_ptr[i], expected[i]))
        {
            printf("  ✗ Mismatch at %d: exp %.1f, got %.1f\n", i, expected[i], out_ptr[i]);
            correct = false;
        }
    }
    
    printf(correct ? "  ✓ PASS\n" : "  ✗ FAIL\n");
    return correct ? 0 : -1;
}

int test_gatherelements_negative()
{
    printf("Testing GatherElements (negative indices)...\n");
    
    // Input: w=3, h=2
    float input_data[] = {1,2,3, 4,5,6};
    ncnn::Mat input = create_float_mat(3, 2, 1, input_data);
    
    // Indices with -1 (last element = 2)
    int index_data[] = {0,-1, -1,0};
    ncnn::Mat indices = create_int_mat(2, 2, 1, index_data);
    
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
    int ret = op->forward(bottom_blobs, top_blobs, opt);
    delete op;
    
    if (ret != 0) { printf("  ✗ Forward failed\n"); return -1; }
    
    // Expected: -1 -> 2 (last index)
    // [0,0]=input[0,0]=1, [1,0]=input[2,0]=3
    // [0,1]=input[2,1]=6, [1,1]=input[0,1]=4
    float expected[] = {1.0f, 3.0f, 6.0f, 4.0f};
    const ncnn::Mat& out = top_blobs[0];
    const float* out_ptr = (const float*)out;
    
    bool correct = true;
    for (int i = 0; i < 4; i++)
    {
        if (!approx_equal(out_ptr[i], expected[i]))
        {
            printf("  ✗ Mismatch at %d: exp %.1f, got %.1f\n", i, expected[i], out_ptr[i]);
            correct = false;
        }
    }
    
    printf(correct ? "  ✓ PASS\n" : "  ✗ FAIL\n");
    return correct ? 0 : -1;
}

// MOD TESTS
int test_mod_basic()
{
    printf("Testing Mod basic...\n");
    
    float a_data[] = {10,11,12,13,14,15,16,17,18,19};
    float b_data[] = {3,3,3,3,3,3,3,3,3,3};
    
    ncnn::Mat a = create_float_mat(10, 1, 1, a_data);
    ncnn::Mat b = create_float_mat(10, 1, 1, b_data);
    
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
    int ret = op->forward(bottom_blobs, top_blobs, opt);
    delete op;
    
    if (ret != 0) { printf("  ✗ Forward failed\n"); return -1; }
    
    float expected[] = {1,2,0,1,2,0,1,2,0,1};
    const ncnn::Mat& out = top_blobs[0];
    const float* out_ptr = (const float*)out;
    
    bool correct = true;
    for (int i = 0; i < 10; i++)
    {
        if (!approx_equal(out_ptr[i], expected[i]))
        {
            printf("  ✗ Mismatch at %d: exp %.1f, got %.1f\n", i, expected[i], out_ptr[i]);
            correct = false;
        }
    }
    
    printf(correct ? "  ✓ PASS\n" : "  ✗ FAIL\n");
    return correct ? 0 : -1;
}

int test_mod_c_style()
{
    printf("Testing Mod (C-style)...\n");
    
    float a_data[] = {-10,-7,-4,-1,2,5,8};
    float b_data[] = {3,3,3,3,3,3,3};
    
    ncnn::Mat a = create_float_mat(7, 1, 1, a_data);
    ncnn::Mat b = create_float_mat(7, 1, 1, b_data);
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::Layer* op = ncnn::create_layer("Mod");
    ncnn::ParamDict pd;
    pd.set(0, 1);
    op->load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = a;
    bottom_blobs[1] = b;
    
    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);
    delete op;
    
    if (ret != 0) { printf("  ✗ Forward failed\n"); return -1; }
    
    float expected[] = {-1,-1,-1,-1,2,2,2};
    const ncnn::Mat& out = top_blobs[0];
    const float* out_ptr = (const float*)out;
    
    bool correct = true;
    for (int i = 0; i < 7; i++)
    {
        if (!approx_equal(out_ptr[i], expected[i]))
        {
            printf("  ✗ Mismatch at %d: exp %.1f, got %.1f\n", i, expected[i], out_ptr[i]);
            correct = false;
        }
    }
    
    printf(correct ? "  ✓ PASS\n" : "  ✗ FAIL\n");
    return correct ? 0 : -1;
}

int test_mod_zero()
{
    printf("Testing Mod (zero divisor)...\n");
    
    float a_data[] = {10,11,12};
    float b_data[] = {0,2,0};
    
    ncnn::Mat a = create_float_mat(3, 1, 1, a_data);
    ncnn::Mat b = create_float_mat(3, 1, 1, b_data);
    
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
    int ret = op->forward(bottom_blobs, top_blobs, opt);
    delete op;
    
    if (ret != 0) { printf("  ✗ Forward failed\n"); return -1; }
    
    float expected[] = {0,1,0};
    const ncnn::Mat& out = top_blobs[0];
    const float* out_ptr = (const float*)out;
    
    bool correct = true;
    for (int i = 0; i < 3; i++)
    {
        if (!approx_equal(out_ptr[i], expected[i]))
        {
            printf("  ✗ Mismatch at %d: exp %.1f, got %.1f\n", i, expected[i], out_ptr[i]);
            correct = false;
        }
    }
    
    printf(correct ? "  ✓ PASS\n" : "  ✗ FAIL\n");
    return correct ? 0 : -1;
}

// TILE TESTS - ncnn uses w x h layout
int test_tile_basic()
{
    printf("Testing Tile basic...\n");
    
    // Input: w=2, h=1
    float input_data[] = {1,2};
    ncnn::Mat input = create_float_mat(2, 1, 1, input_data);
    
    // Repeats: [1, 3] - repeat h by 3
    int repeats_data[] = {1, 3};
    ncnn::Mat repeats = create_int_mat(2, 1, 1, repeats_data);
    
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
    
    if (ret != 0) { printf("  ✗ Forward failed (ret=%d)\n", ret); return -1; }
    
    const ncnn::Mat& out = top_blobs[0];
    if (out.w != 2 || out.h != 3)
    {
        printf("  ✗ Wrong shape: %d x %d (expected 2 x 3)\n", out.w, out.h);
        return -1;
    }
    
    const float* out_ptr = (const float*)out;
    float expected[] = {1,1,1, 2,2,2};
    
    bool correct = true;
    for (int i = 0; i < 6; i++)
    {
        if (!approx_equal(out_ptr[i], expected[i]))
        {
            printf("  ✗ Mismatch at %d: exp %.1f, got %.1f\n", i, expected[i], out_ptr[i]);
            correct = false;
        }
    }
    
    printf(correct ? "  ✓ PASS\n" : "  ✗ FAIL\n");
    return correct ? 0 : -1;
}

int test_tile_1d()
{
    printf("Testing Tile 1D...\n");
    
    // Input: w=3, h=1
    float input_data[] = {1,2,3};
    ncnn::Mat input = create_float_mat(3, 1, 1, input_data);
    
    // Repeats: [2] - repeat w by 2
    int repeats_data[] = {2};
    ncnn::Mat repeats = create_int_mat(1, 1, 1, repeats_data);
    
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
    
    if (ret != 0) { printf("  ✗ Forward failed (ret=%d)\n", ret); return -1; }
    
    const ncnn::Mat& out = top_blobs[0];
    if (out.w != 6 || out.h != 1)
    {
        printf("  ✗ Wrong shape: %d x %d (expected 6 x 1)\n", out.w, out.h);
        return -1;
    }
    
    const float* out_ptr = (const float*)out;
    float expected[] = {1,1,2,2,3,3};
    
    bool correct = true;
    for (int i = 0; i < 6; i++)
    {
        if (!approx_equal(out_ptr[i], expected[i]))
        {
            printf("  ✗ Mismatch at %d: exp %.1f, got %.1f\n", i, expected[i], out_ptr[i]);
            correct = false;
        }
    }
    
    printf(correct ? "  ✓ PASS\n" : "  ✗ FAIL\n");
    return correct ? 0 : -1;
}

// EXPAND TESTS
int test_expand_basic()
{
    printf("Testing Expand basic...\n");
    
    // Input: w=1, h=1
    float input_data[] = {42};
    ncnn::Mat input = create_float_mat(1, 1, 1, input_data);
    
    // Shape: [3] - expand w to 3
    int shape_data[] = {3};
    ncnn::Mat shape = create_int_mat(1, 1, 1, shape_data);
    
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
    
    if (ret != 0) { printf("  ✗ Forward failed (ret=%d)\n", ret); return -1; }
    
    const ncnn::Mat& out = top_blobs[0];
    if (out.w != 3 || out.h != 1)
    {
        printf("  ✗ Wrong shape: %d x %d (expected 3 x 1)\n", out.w, out.h);
        return -1;
    }
    
    const float* out_ptr = (const float*)out;
    
    bool correct = true;
    for (int i = 0; i < 3; i++)
    {
        if (!approx_equal(out_ptr[i], 42.0f))
        {
            printf("  ✗ Mismatch at %d: exp 42.0, got %.1f\n", i, out_ptr[i]);
            correct = false;
        }
    }
    
    printf(correct ? "  ✓ PASS\n" : "  ✗ FAIL\n");
    return correct ? 0 : -1;
}

int test_expand_2d()
{
    printf("Testing Expand 2D...\n");
    
    // Input: w=2, h=1
    float input_data[] = {1,2};
    ncnn::Mat input = create_float_mat(2, 1, 1, input_data);
    
    // Shape: [2, 3] - expand to w=2, h=3
    int shape_data[] = {2, 3};
    ncnn::Mat shape = create_int_mat(2, 1, 1, shape_data);
    
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
    
    if (ret != 0) { printf("  ✗ Forward failed (ret=%d)\n", ret); return -1; }
    
    const ncnn::Mat& out = top_blobs[0];
    if (out.w != 2 || out.h != 3)
    {
        printf("  ✗ Wrong shape: %d x %d (expected 2 x 3)\n", out.w, out.h);
        return -1;
    }
    
    const float* out_ptr = (const float*)out;
    float expected[] = {1,1,1, 2,2,2};
    
    bool correct = true;
    for (int i = 0; i < 6; i++)
    {
        if (!approx_equal(out_ptr[i], expected[i]))
        {
            printf("  ✗ Mismatch at %d: exp %.1f, got %.1f\n", i, expected[i], out_ptr[i]);
            correct = false;
        }
    }
    
    printf(correct ? "  ✓ PASS\n" : "  ✗ FAIL\n");
    return correct ? 0 : -1;
}

int main()
{
    printf("================================================================================\n");
    printf("YOLO26 NCNN Operators - Comprehensive Test Suite\n");
    printf("================================================================================\n\n");
    
    int passed = 0, total = 0;
    
    printf("GATHERELEMENTS TESTS\n");
    printf("--------------------------------------------------------------------------------\n");
    total++; if (test_gatherelements_basic() == 0) passed++;
    total++; if (test_gatherelements_axis1() == 0) passed++;
    total++; if (test_gatherelements_negative() == 0) passed++;
    printf("\n");
    
    printf("MOD TESTS\n");
    printf("--------------------------------------------------------------------------------\n");
    total++; if (test_mod_basic() == 0) passed++;
    total++; if (test_mod_c_style() == 0) passed++;
    total++; if (test_mod_zero() == 0) passed++;
    printf("\n");
    
    printf("TILE TESTS\n");
    printf("--------------------------------------------------------------------------------\n");
    total++; if (test_tile_basic() == 0) passed++;
    total++; if (test_tile_1d() == 0) passed++;
    printf("\n");
    
    printf("EXPAND TESTS\n");
    printf("--------------------------------------------------------------------------------\n");
    total++; if (test_expand_basic() == 0) passed++;
    total++; if (test_expand_2d() == 0) passed++;
    printf("\n");
    
    printf("================================================================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("================================================================================\n");
    
    if (passed == total)
    {
        printf("\n✅ ALL TESTS PASSED!\n");
        return 0;
    }
    else
    {
        printf("\n❌ %d TESTS FAILED\n", total - passed);
        return 1;
    }
}
