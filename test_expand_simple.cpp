// Simple test for Expand operator
#include <stdio.h>
#include "layer/expand.h"
#include "mat.h"
#include "option.h"

int test_expand(int in_w, int in_h, int in_c, int out_w, int out_h, int out_c)
{
    ncnn::Mat input(in_w, in_h, in_c);
    // Fill with test data
    for (int i = 0; i < (int)input.total(); i++)
        ((float*)input)[i] = i + 1.0f;

    // Create shape tensor - should match output dimensions
    int out_dims = 1;
    if (out_h > 1 || out_c > 1) out_dims = 2;
    if (out_c > 1) out_dims = 3;
    
    ncnn::Mat shape_tensor(out_dims);
    int* shape_ptr = (int*)shape_tensor;
    if (out_dims >= 1) shape_ptr[0] = out_w;
    if (out_dims >= 2) shape_ptr[1] = out_h;
    if (out_dims >= 3) shape_ptr[2] = out_c;

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* op = ncnn::create_layer("Expand");

    ncnn::ParamDict pd;
    op->load_param(pd);

    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = input;
    bottom_blobs[1] = shape_tensor;

    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);

    delete op;

    if (ret != 0)
    {
        printf("✗ Expand forward failed\n");
        return -1;
    }

    // Check output shape
    const ncnn::Mat& out = top_blobs[0];
    if (out.w != out_w || out.h != out_h || out.c != out_c)
    {
        printf("✗ Output shape mismatch: expected (%d,%d,%d), got (%d,%d,%d)\n",
                out_w, out_h, out_c, out.w, out.h, out.c);
        return -1;
    }

    printf("✓ PASS: (%d,%d,%d) -> (%d,%d,%d)\n", in_w, in_h, in_c, out_w, out_h, out_c);
    return 0;
}

int main()
{
    printf("================================================================================\n");
    printf("Expand Operator Test\n");
    printf("================================================================================\n\n");

    int passed = 0;
    int total = 0;

    // Test 1: 1D to 1D expansion
    total++; if (test_expand(1, 1, 1, 10, 1, 1) == 0) passed++;
    
    // Test 2: 1D to 2D expansion (broadcasting)
    total++; if (test_expand(5, 1, 1, 5, 3, 1) == 0) passed++;
    
    // Test 3: 2D broadcasting
    total++; if (test_expand(1, 5, 1, 4, 5, 1) == 0) passed++;
    
    // Test 4: 2D to 3D expansion
    total++; if (test_expand(2, 3, 1, 2, 3, 5) == 0) passed++;
    
    // Test 5: 1D to 3D full broadcast
    total++; if (test_expand(1, 1, 1, 4, 6, 8) == 0) passed++;

    printf("\n================================================================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("================================================================================\n");

    if (passed == total)
    {
        printf("\n✅ All Expand tests PASSED!\n");
        return 0;
    }
    else
    {
        printf("\n❌ %d tests FAILED\n", total - passed);
        return 1;
    }
}
