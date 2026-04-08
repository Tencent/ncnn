// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mat.h"
#include "net.h"

#if NCNN_VULKAN
#include "gpu.h"
#include "command.h"
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>

static int test_create_batch_basic()
{
    // create a batch of 4 images, 3 channels, 8x6 spatial
    ncnn::Mat m;
    m.create_batch(8, 6, 3, 4, 4u, 1);

    if (m.dims != 3)
    {
        fprintf(stderr, "test_create_batch_basic dims expect 3 got %d\n", m.dims);
        return -1;
    }
    if (m.w != 8 || m.h != 6 || m.c != 3)
    {
        fprintf(stderr, "test_create_batch_basic shape mismatch w=%d h=%d c=%d\n", m.w, m.h, m.c);
        return -1;
    }
    if (m.n != 4)
    {
        fprintf(stderr, "test_create_batch_basic n expect 4 got %d\n", m.n);
        return -1;
    }
    if (m.data == 0)
    {
        fprintf(stderr, "test_create_batch_basic data is null\n");
        return -1;
    }
    if (m.refcount == 0 || *m.refcount != 1)
    {
        fprintf(stderr, "test_create_batch_basic refcount error\n");
        return -1;
    }

    return 0;
}

static int test_nstep_alignment()
{
    // verify nstep * elemsize is 4K aligned
    {
        ncnn::Mat m;
        m.create_batch(8, 6, 3, 4, 4u, 1);
        size_t nstep_bytes = m.nstep * m.elemsize;
        if (nstep_bytes % 4096 != 0)
        {
            fprintf(stderr, "test_nstep_alignment 3D failed: nstep_bytes=%zu\n", nstep_bytes);
            return -1;
        }
    }

    // odd spatial dims
    {
        ncnn::Mat m;
        m.create_batch(7, 5, 13, 2, 4u, 1);
        size_t nstep_bytes = m.nstep * m.elemsize;
        if (nstep_bytes % 4096 != 0)
        {
            fprintf(stderr, "test_nstep_alignment odd failed: nstep_bytes=%zu\n", nstep_bytes);
            return -1;
        }
    }

    // 4D with depth
    {
        ncnn::Mat m;
        m.create_batch(5, 4, 3, 2, 8, 4u, 1, 0);
        if (m.dims != 4)
        {
            fprintf(stderr, "test_nstep_alignment 4D dims expect 4 got %d\n", m.dims);
            return -1;
        }
        size_t nstep_bytes = m.nstep * m.elemsize;
        if (nstep_bytes % 4096 != 0)
        {
            fprintf(stderr, "test_nstep_alignment 4D failed: nstep_bytes=%zu\n", nstep_bytes);
            return -1;
        }
    }

    // packed elempack=4
    {
        ncnn::Mat m;
        m.create_batch(8, 6, 1, 12, 4, 16u, 4, 0);
        size_t nstep_bytes = m.nstep * m.elemsize;
        if (nstep_bytes % 4096 != 0)
        {
            fprintf(stderr, "test_nstep_alignment packed failed: nstep_bytes=%zu\n", nstep_bytes);
            return -1;
        }
    }

    return 0;
}

static int test_batch_subview_zero_copy()
{
    ncnn::Mat m;
    m.create_batch(4, 3, 2, 3, 4u, 1);

    // fill each batch with distinct value
    for (int b = 0; b < m.n; b++)
    {
        ncnn::Mat sub = m.batch(b);
        sub.fill((float)(b + 1));
    }

    // read back and verify
    for (int b = 0; b < m.n; b++)
    {
        const ncnn::Mat sub = m.batch(b);

        // verify sub-view properties
        if (sub.dims != m.dims || sub.w != m.w || sub.h != m.h || sub.c != m.c)
        {
            fprintf(stderr, "test_batch_subview shape mismatch at batch %d\n", b);
            return -1;
        }
        if (sub.cstep != m.cstep)
        {
            fprintf(stderr, "test_batch_subview cstep mismatch at batch %d\n", b);
            return -1;
        }
        if (sub.n != 1)
        {
            fprintf(stderr, "test_batch_subview n expect 1 got %d\n", sub.n);
            return -1;
        }
        if (sub.refcount != 0)
        {
            fprintf(stderr, "test_batch_subview refcount should be NULL (zero-copy)\n");
            return -1;
        }

        // verify data pointer is at correct offset
        unsigned char* expected_ptr = (unsigned char*)m.data + m.nstep * b * m.elemsize;
        if ((unsigned char*)sub.data != expected_ptr)
        {
            fprintf(stderr, "test_batch_subview data pointer mismatch at batch %d\n", b);
            return -1;
        }

        // verify values
        float expected = (float)(b + 1);
        for (int q = 0; q < sub.c; q++)
        {
            const float* ptr = sub.channel(q);
            for (int i = 0; i < sub.w * sub.h; i++)
            {
                if (ptr[i] != expected)
                {
                    fprintf(stderr, "test_batch_subview value mismatch at batch %d ch %d idx %d: got %f expect %f\n",
                            b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_range()
{
    ncnn::Mat m;
    m.create_batch(4, 3, 2, 4, 4u, 1);

    // fill with batch index
    for (int b = 0; b < 4; b++)
    {
        ncnn::Mat sub = m.batch(b);
        sub.fill((float)(b * 10));
    }

    // get range [1, 2) batches
    ncnn::Mat range = m.batch_range(1, 2);
    if (range.n != 2)
    {
        fprintf(stderr, "test_batch_range n expect 2 got %d\n", range.n);
        return -1;
    }
    if (range.nstep != m.nstep)
    {
        fprintf(stderr, "test_batch_range nstep mismatch\n");
        return -1;
    }

    // verify range.batch(0) == m.batch(1)
    const ncnn::Mat r0 = range.batch(0);
    const float* r0_ptr = r0.channel(0);
    if (r0_ptr[0] != 10.f)
    {
        fprintf(stderr, "test_batch_range batch(0) value expect 10 got %f\n", r0_ptr[0]);
        return -1;
    }

    // verify range.batch(1) == m.batch(2)
    const ncnn::Mat r1 = range.batch(1);
    const float* r1_ptr = r1.channel(0);
    if (r1_ptr[0] != 20.f)
    {
        fprintf(stderr, "test_batch_range batch(1) value expect 20 got %f\n", r1_ptr[0]);
        return -1;
    }

    return 0;
}

static int test_batch_data_isolation()
{
    ncnn::Mat m;
    m.create_batch(16, 16, 3, 4, 4u, 1);

    // write unique pattern to each batch
    for (int b = 0; b < 4; b++)
    {
        ncnn::Mat sub = m.batch(b);
        for (int q = 0; q < sub.c; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < sub.w * sub.h; i++)
            {
                ptr[i] = (float)(b * 1000 + q * 100 + i);
            }
        }
    }

    // verify no cross-contamination
    for (int b = 0; b < 4; b++)
    {
        const ncnn::Mat sub = m.batch(b);
        for (int q = 0; q < sub.c; q++)
        {
            const float* ptr = sub.channel(q);
            for (int i = 0; i < sub.w * sub.h; i++)
            {
                float expected = (float)(b * 1000 + q * 100 + i);
                if (ptr[i] != expected)
                {
                    fprintf(stderr, "test_batch_data_isolation mismatch at b=%d q=%d i=%d: got %f expect %f\n",
                            b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_clone()
{
    ncnn::Mat m;
    m.create_batch(8, 6, 3, 4, 4u, 1);

    // fill with data
    for (int b = 0; b < 4; b++)
    {
        ncnn::Mat sub = m.batch(b);
        sub.fill((float)(b + 1));
    }

    // clone
    ncnn::Mat m2 = m.clone();

    // verify deep copy
    if (m2.data == m.data)
    {
        fprintf(stderr, "test_batch_clone data should be different (deep copy)\n");
        return -1;
    }
    if (m2.n != m.n)
    {
        fprintf(stderr, "test_batch_clone n mismatch\n");
        return -1;
    }
    if (m2.nstep != m.nstep)
    {
        fprintf(stderr, "test_batch_clone nstep mismatch\n");
        return -1;
    }
    if (m2.dims != m.dims || m2.w != m.w || m2.h != m.h || m2.c != m.c)
    {
        fprintf(stderr, "test_batch_clone shape mismatch\n");
        return -1;
    }

    // verify values match
    for (int b = 0; b < 4; b++)
    {
        const ncnn::Mat s2 = m2.batch(b);
        float expected = (float)(b + 1);
        const float* p2 = s2.channel(0);
        if (p2[0] != expected)
        {
            fprintf(stderr, "test_batch_clone value mismatch at batch %d\n", b);
            return -1;
        }
    }

    // verify independence: modify original, clone should not change
    m.batch(0).fill(999.f);
    const float* p2 = m2.batch(0).channel(0);
    if (p2[0] != 1.f)
    {
        fprintf(stderr, "test_batch_clone not independent after modify\n");
        return -1;
    }

    return 0;
}

static int test_batch_release()
{
    ncnn::Mat m;
    m.create_batch(4, 3, 2, 4, 4u, 1);

    m.release();

    if (m.dims != 0)
    {
        fprintf(stderr, "test_batch_release dims expect 0 got %d\n", m.dims);
        return -1;
    }
    if (m.n != 1)
    {
        fprintf(stderr, "test_batch_release n expect 1 got %d\n", m.n);
        return -1;
    }
    if (m.nstep != 0)
    {
        fprintf(stderr, "test_batch_release nstep expect 0 got %zu\n", m.nstep);
        return -1;
    }
    if (m.data != 0)
    {
        fprintf(stderr, "test_batch_release data should be null\n");
        return -1;
    }

    return 0;
}

static int test_backward_compatibility()
{
    // regular Mat should have n=1
    ncnn::Mat m1(8, 6, 3);
    if (m1.n != 1)
    {
        fprintf(stderr, "test_backward_compat n expect 1 got %d\n", m1.n);
        return -1;
    }

    // channel() and row() still work
    m1.fill(42.f);
    ncnn::Mat ch0 = m1.channel(0);
    if (ch0.w != 8 || ch0.h != 6)
    {
        fprintf(stderr, "test_backward_compat channel shape mismatch\n");
        return -1;
    }
    const float* row0 = ch0.row(0);
    if (row0[0] != 42.f)
    {
        fprintf(stderr, "test_backward_compat channel value mismatch\n");
        return -1;
    }

    // copy ctor preserves n
    ncnn::Mat m2 = m1;
    if (m2.n != 1)
    {
        fprintf(stderr, "test_backward_compat copy n mismatch\n");
        return -1;
    }

    // empty Mat has n=1
    ncnn::Mat m3;
    if (m3.n != 1)
    {
        fprintf(stderr, "test_backward_compat empty n expect 1 got %d\n", m3.n);
        return -1;
    }

    return 0;
}

static int test_create_batch_single()
{
    // create_batch with batch=1 should fall back to regular create
    ncnn::Mat m;
    m.create_batch(8, 6, 3, 1, 4u, 1);

    if (m.dims != 3)
    {
        fprintf(stderr, "test_create_batch_single dims expect 3 got %d\n", m.dims);
        return -1;
    }
    if (m.n != 1)
    {
        fprintf(stderr, "test_create_batch_single n expect 1 got %d\n", m.n);
        return -1;
    }
    if (m.w != 8 || m.h != 6 || m.c != 3)
    {
        fprintf(stderr, "test_create_batch_single shape mismatch\n");
        return -1;
    }

    // should work like normal Mat
    m.fill(7.f);
    if (((const float*)m.data)[0] != 7.f)
    {
        fprintf(stderr, "test_create_batch_single fill failed\n");
        return -1;
    }

    return 0;
}

static int test_create_batch_1d()
{
    // create a batch of 4 1D vectors, w=100
    ncnn::Mat m;
    m.create_batch(100, 4, 4u, 1);

    if (m.dims != 1)
    {
        fprintf(stderr, "test_create_batch_1d dims expect 1 got %d\n", m.dims);
        return -1;
    }
    if (m.w != 100 || m.h != 1 || m.d != 1 || m.c != 1)
    {
        fprintf(stderr, "test_create_batch_1d shape mismatch w=%d h=%d d=%d c=%d\n", m.w, m.h, m.d, m.c);
        return -1;
    }
    if (m.n != 4)
    {
        fprintf(stderr, "test_create_batch_1d n expect 4 got %d\n", m.n);
        return -1;
    }
    if (m.data == 0)
    {
        fprintf(stderr, "test_create_batch_1d data is null\n");
        return -1;
    }

    // verify nstep alignment
    size_t nstep_bytes = m.nstep * m.elemsize;
    if (nstep_bytes % 4096 != 0)
    {
        fprintf(stderr, "test_create_batch_1d nstep_bytes=%zu not 4K aligned\n", nstep_bytes);
        return -1;
    }

    // fill and verify subview zero-copy
    for (int b = 0; b < m.n; b++)
    {
        ncnn::Mat sub = m.batch(b);
        sub.fill((float)(b + 10));
    }
    for (int b = 0; b < m.n; b++)
    {
        const ncnn::Mat sub = m.batch(b);
        if (sub.dims != 1 || sub.w != 100 || sub.n != 1)
        {
            fprintf(stderr, "test_create_batch_1d subview shape mismatch at batch %d\n", b);
            return -1;
        }
        if (sub.refcount != 0)
        {
            fprintf(stderr, "test_create_batch_1d subview should be zero-copy\n");
            return -1;
        }
        float expected = (float)(b + 10);
        const float* ptr = (const float*)sub.data;
        if (ptr[0] != expected || ptr[99] != expected)
        {
            fprintf(stderr, "test_create_batch_1d value mismatch at batch %d\n", b);
            return -1;
        }
    }

    return 0;
}

static int test_create_batch_2d()
{
    // create a batch of 3 2D matrices, 10x20
    ncnn::Mat m;
    m.create_batch(10, 20, 3, 4u, 1);

    if (m.dims != 2)
    {
        fprintf(stderr, "test_create_batch_2d dims expect 2 got %d\n", m.dims);
        return -1;
    }
    if (m.w != 10 || m.h != 20 || m.d != 1 || m.c != 1)
    {
        fprintf(stderr, "test_create_batch_2d shape mismatch w=%d h=%d d=%d c=%d\n", m.w, m.h, m.d, m.c);
        return -1;
    }
    if (m.n != 3)
    {
        fprintf(stderr, "test_create_batch_2d n expect 3 got %d\n", m.n);
        return -1;
    }

    // verify nstep alignment
    size_t nstep_bytes = m.nstep * m.elemsize;
    if (nstep_bytes % 4096 != 0)
    {
        fprintf(stderr, "test_create_batch_2d nstep_bytes=%zu not 4K aligned\n", nstep_bytes);
        return -1;
    }

    // fill and verify subview zero-copy
    for (int b = 0; b < m.n; b++)
    {
        ncnn::Mat sub = m.batch(b);
        sub.fill((float)(b + 100));
    }
    for (int b = 0; b < m.n; b++)
    {
        const ncnn::Mat sub = m.batch(b);
        if (sub.dims != 2 || sub.w != 10 || sub.h != 20 || sub.n != 1)
        {
            fprintf(stderr, "test_create_batch_2d subview shape mismatch at batch %d\n", b);
            return -1;
        }
        if (sub.refcount != 0)
        {
            fprintf(stderr, "test_create_batch_2d subview should be zero-copy\n");
            return -1;
        }
        float expected = (float)(b + 100);
        const float* ptr = (const float*)sub.data;
        if (ptr[0] != expected || ptr[10 * 20 - 1] != expected)
        {
            fprintf(stderr, "test_create_batch_2d value mismatch at batch %d\n", b);
            return -1;
        }
    }

    return 0;
}

static int test_batch_forward_relu()
{
    // Build a minimal Input -> ReLU network
    // ReLU with slope=0.1 (leaky relu)
    const char param_str[] =
        "7767517\n"
        "2 2\n"
        "Input input 0 1 data\n"
        "ReLU  relu  1 1 data output 0=1.000000e-01\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 4;
    const int C = 3;
    const int H = 3;
    const int W = 4;

    ncnn::Mat input_batch;
    input_batch.create_batch(W, H, C, B, 4u, 1);
    if (input_batch.empty())
    {
        fprintf(stderr, "test_batch_forward_relu create_batch failed\n");
        return -1;
    }

    // fill: batch b gets value (b - 1.5), some negative, some positive
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        sub.fill((float)(b - 1.5f));
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output_batch;
    int ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_relu extract failed ret=%d\n", ret);
        return -1;
    }

    if (output_batch.n != B)
    {
        fprintf(stderr, "test_batch_forward_relu output n expect %d got %d\n", B, output_batch.n);
        return -1;
    }
    if (output_batch.w != W || output_batch.h != H || output_batch.c != C)
    {
        fprintf(stderr, "test_batch_forward_relu output shape mismatch\n");
        return -1;
    }

    // verify leaky relu: max(x, 0.1*x)
    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat out_sub = output_batch.batch(b);
        float input_val = (float)(b - 1.5f);
        float expected = input_val > 0 ? input_val : input_val * 0.1f;

        for (int q = 0; q < C; q++)
        {
            const float* ptr = out_sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                if (fabsf(ptr[i] - expected) > 1e-5f)
                {
                    fprintf(stderr, "test_batch_forward_relu value mismatch at b=%d q=%d i=%d: got %f expect %f\n",
                            b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_forward_pooling()
{
    // Input -> Pooling(max, 2x2, stride=2)
    const char param_str[] =
        "7767517\n"
        "2 2\n"
        "Input   input   0 1 data\n"
        "Pooling pooling 1 1 data output 0=0 1=2 2=2\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 2;
    const int C = 2;
    const int H = 4;
    const int W = 4;

    ncnn::Mat input_batch;
    input_batch.create_batch(W, H, C, B, 4u, 1);

    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 10 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output_batch;
    int ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_pooling extract failed ret=%d\n", ret);
        return -1;
    }

    if (output_batch.n != B)
    {
        fprintf(stderr, "test_batch_forward_pooling output n expect %d got %d\n", B, output_batch.n);
        return -1;
    }
    if (output_batch.w != 2 || output_batch.h != 2 || output_batch.c != C)
    {
        fprintf(stderr, "test_batch_forward_pooling output shape expect 2x2x%d got %dx%dx%d\n",
                C, output_batch.w, output_batch.h, output_batch.c);
        return -1;
    }

    // verify max pooling for batch 0, channel 0
    // input 4x4: [ 0  1  2  3 / 4  5  6  7 / 8  9 10 11 / 12 13 14 15 ]
    // max pool 2x2 stride 2 -> [ 5 7 / 13 15 ]
    {
        const ncnn::Mat out0 = output_batch.batch(0);
        const float* ptr = out0.channel(0);
        float expected[4] = {5.f, 7.f, 13.f, 15.f};
        for (int i = 0; i < 4; i++)
        {
            if (fabsf(ptr[i] - expected[i]) > 1e-5f)
            {
                fprintf(stderr, "test_batch_forward_pooling b0 mismatch at i=%d: got %f expect %f\n",
                        i, ptr[i], expected[i]);
                return -1;
            }
        }
    }

    // verify batch 1, channel 0: input 100+i -> max pool -> [105, 107, 113, 115]
    {
        const ncnn::Mat out1 = output_batch.batch(1);
        const float* ptr = out1.channel(0);
        float expected[4] = {105.f, 107.f, 113.f, 115.f};
        for (int i = 0; i < 4; i++)
        {
            if (fabsf(ptr[i] - expected[i]) > 1e-5f)
            {
                fprintf(stderr, "test_batch_forward_pooling b1 mismatch at i=%d: got %f expect %f\n",
                        i, ptr[i], expected[i]);
                return -1;
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
static int test_vkmat_create_batch_basic()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();

    ncnn::VkMat m;
    m.create_batch(8, 6, 3, 4, 4u, 1, blob_allocator);

    if (m.dims != 3)
    {
        fprintf(stderr, "test_vkmat_create_batch_basic dims expect 3 got %d\n", m.dims);
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }
    if (m.w != 8 || m.h != 6 || m.c != 3)
    {
        fprintf(stderr, "test_vkmat_create_batch_basic shape mismatch\n");
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }
    if (m.n != 4)
    {
        fprintf(stderr, "test_vkmat_create_batch_basic n expect 4 got %d\n", m.n);
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }
    if (m.data == 0)
    {
        fprintf(stderr, "test_vkmat_create_batch_basic data is null\n");
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    m.release();
    vkdev->reclaim_blob_allocator(blob_allocator);
    return 0;
}

static int test_vkmat_nstep_alignment()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();

    ncnn::VkMat m;
    m.create_batch(7, 5, 13, 4, 4u, 1, blob_allocator);

    size_t nstep_bytes = m.nstep * m.elemsize;
    if (nstep_bytes % 4096 != 0)
    {
        fprintf(stderr, "test_vkmat_nstep_alignment failed: nstep_bytes=%zu\n", nstep_bytes);
        m.release();
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    m.release();
    vkdev->reclaim_blob_allocator(blob_allocator);
    return 0;
}

static int test_vkmat_batch_subview()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();

    ncnn::VkMat m;
    m.create_batch(4, 3, 2, 3, 4u, 1, blob_allocator);

    for (int b = 0; b < m.n; b++)
    {
        const ncnn::VkMat sub = m.batch(b);

        // verify sub-view properties
        if (sub.dims != m.dims || sub.w != m.w || sub.h != m.h || sub.c != m.c)
        {
            fprintf(stderr, "test_vkmat_batch_subview shape mismatch at batch %d\n", b);
            m.release();
            vkdev->reclaim_blob_allocator(blob_allocator);
            return -1;
        }
        if (sub.cstep != m.cstep)
        {
            fprintf(stderr, "test_vkmat_batch_subview cstep mismatch at batch %d\n", b);
            m.release();
            vkdev->reclaim_blob_allocator(blob_allocator);
            return -1;
        }
        if (sub.n != 1)
        {
            fprintf(stderr, "test_vkmat_batch_subview n expect 1 got %d\n", sub.n);
            m.release();
            vkdev->reclaim_blob_allocator(blob_allocator);
            return -1;
        }
        if (sub.refcount != 0)
        {
            fprintf(stderr, "test_vkmat_batch_subview refcount should be NULL (zero-copy)\n");
            m.release();
            vkdev->reclaim_blob_allocator(blob_allocator);
            return -1;
        }

        // verify buffer_offset is correct
        size_t expected_offset = m.buffer_offset() + m.nstep * b * m.elemsize;
        if (sub.buffer_offset() != expected_offset)
        {
            fprintf(stderr, "test_vkmat_batch_subview buffer_offset mismatch at batch %d: got %zu expect %zu\n",
                    b, sub.buffer_offset(), expected_offset);
            m.release();
            vkdev->reclaim_blob_allocator(blob_allocator);
            return -1;
        }

        // verify same underlying VkBuffer
        if (sub.buffer() != m.buffer())
        {
            fprintf(stderr, "test_vkmat_batch_subview buffer handle mismatch at batch %d\n", b);
            m.release();
            vkdev->reclaim_blob_allocator(blob_allocator);
            return -1;
        }
    }

    m.release();
    vkdev->reclaim_blob_allocator(blob_allocator);
    return 0;
}

static int test_vkmat_batch_range()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();

    ncnn::VkMat m;
    m.create_batch(4, 3, 2, 4, 4u, 1, blob_allocator);

    ncnn::VkMat range = m.batch_range(1, 2);
    if (range.n != 2)
    {
        fprintf(stderr, "test_vkmat_batch_range n expect 2 got %d\n", range.n);
        m.release();
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }
    if (range.nstep != m.nstep)
    {
        fprintf(stderr, "test_vkmat_batch_range nstep mismatch\n");
        m.release();
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    // verify range.batch(0) buffer_offset == m.batch(1) buffer_offset
    if (range.batch(0).buffer_offset() != m.batch(1).buffer_offset())
    {
        fprintf(stderr, "test_vkmat_batch_range offset mismatch at range batch 0\n");
        m.release();
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    // verify range.batch(1) buffer_offset == m.batch(2) buffer_offset
    if (range.batch(1).buffer_offset() != m.batch(2).buffer_offset())
    {
        fprintf(stderr, "test_vkmat_batch_range offset mismatch at range batch 1\n");
        m.release();
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    m.release();
    vkdev->reclaim_blob_allocator(blob_allocator);
    return 0;
}

static int test_vkmat_batch_release()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();

    ncnn::VkMat m;
    m.create_batch(4, 3, 2, 4, 4u, 1, blob_allocator);
    m.release();

    if (m.dims != 0)
    {
        fprintf(stderr, "test_vkmat_batch_release dims expect 0 got %d\n", m.dims);
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }
    if (m.n != 1)
    {
        fprintf(stderr, "test_vkmat_batch_release n expect 1 got %d\n", m.n);
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    return 0;
}

static int test_vkmat_batch_upload_download()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    const int B = 3;
    const int W = 4;
    const int H = 3;
    const int C = 2;

    // create and fill cpu batch
    ncnn::Mat cpu_batch;
    cpu_batch.create_batch(W, H, C, B, 4u, 1);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = cpu_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 10 + i);
            }
        }
    }

    // upload each batch, assemble on gpu, download back
    ncnn::VkCompute cmd(vkdev);

    ncnn::Option opt;
    opt.blob_vkallocator = blob_allocator;
    opt.workspace_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;
    opt.use_vulkan_compute = true;

    ncnn::VkMat gpu_batch;
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat cpu_b = cpu_batch.batch(b);
        ncnn::VkMat gpu_b;
        cmd.record_upload(cpu_b, gpu_b, opt);

        if (b == 0)
        {
            gpu_batch.create_like_batch(gpu_b, B, blob_allocator);
        }

        ncnn::VkMat gpu_batch_slot = gpu_batch.batch(b);
        cmd.record_clone(gpu_b, gpu_batch_slot, opt);
    }

    // download each batch back
    std::vector<ncnn::Mat> cpu_results(B);
    for (int b = 0; b < B; b++)
    {
        ncnn::VkMat gpu_b = gpu_batch.batch(b);
        cmd.record_download(gpu_b, cpu_results[b], opt);
    }

    int ret = cmd.submit_and_wait();
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_upload_download submit failed ret=%d\n", ret);
        vkdev->reclaim_staging_allocator(staging_allocator);
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    // verify downloaded data matches original
    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat& result = cpu_results[b];
        if (result.w != W || result.h != H || result.c != C)
        {
            fprintf(stderr, "test_vkmat_batch_upload_download shape mismatch at batch %d\n", b);
            vkdev->reclaim_staging_allocator(staging_allocator);
            vkdev->reclaim_blob_allocator(blob_allocator);
            return -1;
        }

        const ncnn::Mat orig = cpu_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            const float* orig_ptr = orig.channel(q);
            const float* result_ptr = result.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                if (fabsf(orig_ptr[i] - result_ptr[i]) > 1e-5f)
                {
                    fprintf(stderr, "test_vkmat_batch_upload_download value mismatch at b=%d q=%d i=%d: got %f expect %f\n",
                            b, q, i, result_ptr[i], orig_ptr[i]);
                    vkdev->reclaim_staging_allocator(staging_allocator);
                    vkdev->reclaim_blob_allocator(blob_allocator);
                    return -1;
                }
            }
        }
    }

    vkdev->reclaim_staging_allocator(staging_allocator);
    vkdev->reclaim_blob_allocator(blob_allocator);
    return 0;
}

static int test_vkmat_batch_forward_relu()
{
    const char param_str[] =
        "7767517\n"
        "2 2\n"
        "Input input 0 1 data\n"
        "ReLU  relu  1 1 data output 0=1.000000e-01\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    const int B = 4;
    const int C = 3;
    const int H = 3;
    const int W = 4;

    ncnn::Mat input_batch;
    input_batch.create_batch(W, H, C, B, 4u, 1);
    if (input_batch.empty())
    {
        fprintf(stderr, "test_vkmat_batch_forward_relu create_batch failed\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        sub.fill((float)(b - 1.5f));
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output_batch;
    int ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_relu extract failed ret=%d\n", ret);
        return -1;
    }

    if (output_batch.n != B)
    {
        fprintf(stderr, "test_vkmat_batch_forward_relu output n expect %d got %d\n", B, output_batch.n);
        return -1;
    }
    if (output_batch.w != W || output_batch.h != H || output_batch.c != C)
    {
        fprintf(stderr, "test_vkmat_batch_forward_relu output shape mismatch\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat out_sub = output_batch.batch(b);
        float input_val = (float)(b - 1.5f);
        float expected = input_val > 0 ? input_val : input_val * 0.1f;

        for (int q = 0; q < C; q++)
        {
            const float* ptr = out_sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                if (fabsf(ptr[i] - expected) > 1e-4f)
                {
                    fprintf(stderr, "test_vkmat_batch_forward_relu value mismatch at b=%d q=%d i=%d: got %f expect %f\n",
                            b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_pooling()
{
    const char param_str[] =
        "7767517\n"
        "2 2\n"
        "Input   input   0 1 data\n"
        "Pooling pooling 1 1 data output 0=0 1=2 2=2\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    const int B = 2;
    const int C = 2;
    const int H = 4;
    const int W = 4;

    ncnn::Mat input_batch;
    input_batch.create_batch(W, H, C, B, 4u, 1);

    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 10 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output_batch;
    int ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_pooling extract failed ret=%d\n", ret);
        return -1;
    }

    if (output_batch.n != B)
    {
        fprintf(stderr, "test_vkmat_batch_forward_pooling output n expect %d got %d\n", B, output_batch.n);
        return -1;
    }
    if (output_batch.w != 2 || output_batch.h != 2 || output_batch.c != C)
    {
        fprintf(stderr, "test_vkmat_batch_forward_pooling output shape expect 2x2x%d got %dx%dx%d\n",
                C, output_batch.w, output_batch.h, output_batch.c);
        return -1;
    }

    // verify max pooling for batch 0, channel 0
    // input 4x4: [ 0  1  2  3 / 4  5  6  7 / 8  9 10 11 / 12 13 14 15 ]
    // max pool 2x2 stride 2 -> [ 5 7 / 13 15 ]
    {
        const ncnn::Mat out0 = output_batch.batch(0);
        const float* ptr = out0.channel(0);
        float expected[4] = {5.f, 7.f, 13.f, 15.f};
        for (int i = 0; i < 4; i++)
        {
            if (fabsf(ptr[i] - expected[i]) > 1e-4f)
            {
                fprintf(stderr, "test_vkmat_batch_forward_pooling b0 mismatch at i=%d: got %f expect %f\n",
                        i, ptr[i], expected[i]);
                return -1;
            }
        }
    }

    // verify batch 1, channel 0: input 100+i -> max pool -> [105, 107, 113, 115]
    {
        const ncnn::Mat out1 = output_batch.batch(1);
        const float* ptr = out1.channel(0);
        float expected[4] = {105.f, 107.f, 113.f, 115.f};
        for (int i = 0; i < 4; i++)
        {
            if (fabsf(ptr[i] - expected[i]) > 1e-4f)
            {
                fprintf(stderr, "test_vkmat_batch_forward_pooling b1 mismatch at i=%d: got %f expect %f\n",
                        i, ptr[i], expected[i]);
                return -1;
            }
        }
    }

    return 0;
}
#endif // NCNN_VULKAN

int main()
{
    int ret = 0;

    ret |= test_create_batch_basic();
    ret |= test_nstep_alignment();
    ret |= test_batch_subview_zero_copy();
    ret |= test_batch_range();
    ret |= test_batch_data_isolation();
    ret |= test_batch_clone();
    ret |= test_batch_release();
    ret |= test_backward_compatibility();
    ret |= test_create_batch_single();
    ret |= test_create_batch_1d();
    ret |= test_create_batch_2d();
    ret |= test_batch_forward_relu();
    ret |= test_batch_forward_pooling();

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    if (ncnn::get_gpu_count() > 0)
    {
        ret |= test_vkmat_create_batch_basic();
        ret |= test_vkmat_nstep_alignment();
        ret |= test_vkmat_batch_subview();
        ret |= test_vkmat_batch_range();
        ret |= test_vkmat_batch_release();
        ret |= test_vkmat_batch_upload_download();
        ret |= test_vkmat_batch_forward_relu();
        ret |= test_vkmat_batch_forward_pooling();
    }
    else
    {
        fprintf(stderr, "no vulkan device, skip vkmat batch tests\n");
    }
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    if (ret == 0)
        fprintf(stderr, "test_mat_batch passed\n");

    return ret;
}
