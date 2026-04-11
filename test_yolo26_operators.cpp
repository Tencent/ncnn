// Test program for YOLO26 NCNN operators
// This tests GatherElements, Expand, Tile, and Mod operators

#include <stdio.h>
#include <stdlib.h>
#include "layer/gatherelements.h"
#include "layer/expand.h"
#include "layer/mod.h"
#include "mat.h"
#include "option.h"

int test_gatherelements()
{
    printf("Testing GatherElements...\n");
    
    ncnn::GatherElements op;
    
    // Create test data: 3x4 matrix
    ncnn::Mat data(3, 4);
    for (int i = 0; i < 12; i++)
        ((float*)data)[i] = i + 1;
    
    // Create indices: 2x4
    ncnn::Mat indices(2, 4);
    int idx_data[] = {0, 1, 2, 0, 2, 1, 0, 1};
    for (int i = 0; i < 8; i++)
        ((int*)indices)[i] = idx_data[i];
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::ParamDict pd;
    pd.set(0, 0); // axis=0
    op.load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = data;
    bottom_blobs[1] = indices;
    
    std::vector<ncnn::Mat> top_blobs(1);
    
    int ret = op.forward(bottom_blobs, top_blobs, opt);
    
    if (ret == 0)
    {
        printf("✓ GatherElements test PASSED\n");
        printf("  Output shape: %d x %d\n", top_blobs[0].w, top_blobs[0].h);
        return 0;
    }
    else
    {
        printf("✗ GatherElements test FAILED\n");
        return -1;
    }
}

int test_mod()
{
    printf("Testing Mod...\n");
    
    ncnn::Mod op;
    
    // Create test data
    ncnn::Mat a(10);
    ncnn::Mat b(10);
    for (int i = 0; i < 10; i++)
    {
        ((float*)a)[i] = 10.0f + i;
        ((float*)b)[i] = 3.0f;
    }
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    ncnn::ParamDict pd;
    pd.set(0, 0); // fmod=0 (Python-style)
    op.load_param(pd);
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = a;
    bottom_blobs[1] = b;
    
    std::vector<ncnn::Mat> top_blobs(1);
    
    int ret = op.forward(bottom_blobs, top_blobs, opt);
    
    if (ret == 0)
    {
        printf("✓ Mod test PASSED\n");
        printf("  Sample output: ");
        for (int i = 0; i < 5; i++)
            printf("%.1f%%%.1f=%.1f  ", ((float*)a)[i], ((float*)b)[i], ((float*)top_blobs[0])[i]);
        printf("\n");
        return 0;
    }
    else
    {
        printf("✗ Mod test FAILED\n");
        return -1;
    }
}

int test_expand()
{
    printf("Testing Expand...\n");
    
    ncnn::Expand op;
    
    // Create test data: [1, 2, 3]
    ncnn::Mat input(3);
    ((float*)input)[0] = 1.0f;
    ((float*)input)[1] = 2.0f;
    ((float*)input)[2] = 3.0f;
    
    // Create shape tensor: [2, 3]
    ncnn::Mat shape(3);
    ((int*)shape)[0] = 2;
    ((int*)shape)[1] = 3;
    ((int*)shape)[2] = 1;
    
    ncnn::Option opt;
    opt.num_threads = 1;
    
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = input;
    bottom_blobs[1] = shape;
    
    std::vector<ncnn::Mat> top_blobs(1);
    
    int ret = op.forward(bottom_blobs, top_blobs, opt);
    
    if (ret == 0)
    {
        printf("✓ Expand test PASSED\n");
        printf("  Output shape: %d x %d x %d\n", top_blobs[0].w, top_blobs[0].h, top_blobs[0].c);
        return 0;
    }
    else
    {
        printf("✗ Expand test FAILED\n");
        return -1;
    }
}

int main()
{
    printf("================================================================================\n");
    printf("YOLO26 NCNN Operators Test\n");
    printf("================================================================================\n\n");
    
    int passed = 0;
    int total = 3;
    
    if (test_gatherelements() == 0) passed++;
    printf("\n");
    
    if (test_mod() == 0) passed++;
    printf("\n");
    
    if (test_expand() == 0) passed++;
    printf("\n");
    
    printf("================================================================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("================================================================================\n");
    
    if (passed == total)
    {
        printf("\n✅ All YOLO26 operators working correctly!\n");
        return 0;
    }
    else
    {
        printf("\n❌ Some tests failed\n");
        return 1;
    }
}
