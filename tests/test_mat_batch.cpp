// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mat.h"
#include "net.h"
#include "testutil.h"

#if NCNN_VULKAN
#include "gpu.h"
#include "command.h"
#endif

#include <stdio.h>
#include <string.h>

static int test_create_batch_basic()
{
    // create a batch of 4 images, 3 channels, 8x6 spatial
    ncnn::Mat m;
    m.create(8, 6, 3, 4u, 1, 4);

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
        m.create(8, 6, 3, 4u, 1, 4);
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
        m.create(7, 5, 13, 4u, 1, 2);
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
        m.create(5, 4, 3, 2, 4u, 1, 8, 0);
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
        m.create(8, 6, 1, 12, 16u, 4, 4, 0);
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
    m.create(4, 3, 2, 4u, 1, 3);

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
    m.create(4, 3, 2, 4u, 1, 4);

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
    m.create(16, 16, 3, 4u, 1, 4);

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
    m.create(8, 6, 3, 4u, 1, 4);

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
    m.create(4, 3, 2, 4u, 1, 4);

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

static int test_batch_create_reset()
{
    ncnn::Mat m;
    m.create(4, 3, 2, 4u, 1, 3);
    m.create(4, 3, 2, (size_t)4u, (ncnn::Allocator*)0);

    if (m.n != 1)
    {
        fprintf(stderr, "test_batch_create_reset n expect 1 got %d\n", m.n);
        return -1;
    }
    if (m.nstep != m.total())
    {
        fprintf(stderr, "test_batch_create_reset nstep expect %zu got %zu\n", m.total(), m.nstep);
        return -1;
    }
    if (m.dims != 3 || m.w != 4 || m.h != 3 || m.c != 2)
    {
        fprintf(stderr, "test_batch_create_reset shape mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape()
{
    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 2;

    ncnn::Mat m;
    m.create(W, H, C, 4u, 1, B);

    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = m.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 10 + i);
            }
        }
    }

    ncnn::Mat r = m.reshape(W * H * C);
    if (r.n != B)
    {
        fprintf(stderr, "test_batch_reshape n expect %d got %d\n", B, r.n);
        return -1;
    }
    if (r.dims != 1 || r.w != W * H * C)
    {
        fprintf(stderr, "test_batch_reshape shape mismatch dims=%d w=%d\n", r.dims, r.w);
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat sub = r.batch(b);
        const float* ptr = sub;
        for (int i = 0; i < W * H * C; i++)
        {
            float expected = (float)(b * 100 + (i / (W * H)) * 10 + i % (W * H));
            if (!NearlyEqual(ptr[i], expected, 1e-5f))
            {
                fprintf(stderr, "test_batch_reshape mismatch at b=%d i=%d got %f expect %f\n", b, i, ptr[i], expected);
                return -1;
            }
        }
    }

    return 0;
}

static int test_batch_reshape_zero_copy()
{
    const int B = 3;
    const int C = 2;
    const int H = 4;
    const int W = 4;

    ncnn::Mat m;
    m.create(W, H, C, 4u, 1, B);

    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = m.batch(b);
        float* ptr = sub;
        for (int i = 0; i < W * H * C; i++)
        {
            ptr[i] = (float)(b * 100 + i);
        }
    }

    ncnn::Mat r = m.reshape(W * H * C);
    if (r.data != m.data || r.nstep != m.nstep)
    {
        fprintf(stderr, "test_batch_reshape_zero_copy should share batch storage\n");
        return -1;
    }
    if (r.n != B || r.dims != 1 || r.w != W * H * C)
    {
        fprintf(stderr, "test_batch_reshape_zero_copy shape mismatch\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat sub = r.batch(b);
        const float* ptr = sub;
        for (int i = 0; i < W * H * C; i++)
        {
            float expected = (float)(b * 100 + i);
            if (!NearlyEqual(ptr[i], expected, 1e-5f))
            {
                fprintf(stderr, "test_batch_reshape_zero_copy mismatch at b=%d i=%d got %f expect %f\n", b, i, ptr[i], expected);
                return -1;
            }
        }
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_flatten()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=-1 12=0 13=233\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    int ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_flatten extract failed ret=%d\n", ret);
        return -1;
    }
    if (output.n != 1 || output.dims != 1 || output.w != B * C * H * W)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_flatten shape mismatch\n");
        return -1;
    }

    const float* ptr = output;
    for (int b = 0; b < B; b++)
    {
        for (int q = 0; q < C; q++)
        {
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(b * 100 + q * 20 + i);
                if (!NearlyEqual(*ptr, expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_reshape_batch_to_dim_flatten mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, *ptr, expected);
                    return -1;
                }
                ptr++;
            }
        }
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_4d()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=-1 12=0 13=233\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    int ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_4d extract failed ret=%d\n", ret);
        return -1;
    }
    if (output.n != 1 || output.dims != 4 || output.w != W || output.h != H || output.d != C || output.c != B)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_4d shape mismatch\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat sub = output.channel(b);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(b * 100 + q * 20 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_reshape_batch_to_dim_4d mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_negative_axis()
{
    const char param_str_ref[] = "7767517\n"
                                 "2 2\n"
                                 "Input   input   0 1 data\n"
                                 "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=-1 12=0 13=233\n";

    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=-1 12=-4 13=233\n";

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.load_param_mem(param_str_ref);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_negative_axis reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_negative_axis extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_negative_axis value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_shape_expr()
{
    const char param_str_ref[] = "7767517\n"
                                 "2 2\n"
                                 "Input   input   0 1 data\n"
                                 "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=3 12=0 13=233\n";

    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 6=\"0w,0h,0c,0n\" 12=0 13=233\n";

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.load_param_mem(param_str_ref);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_shape_expr reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_shape_expr extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_shape_expr value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_dim_to_batch()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=3 12=233 13=0\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input;
    input.create(W, H, C, B, 4u, 1);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input.channel(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output_batch;
    int ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch extract failed ret=%d\n", ret);
        return -1;
    }
    if (output_batch.n != B || output_batch.dims != 3 || output_batch.w != W || output_batch.h != H || output_batch.c != C)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch shape mismatch\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat sub = output_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(b * 100 + q * 20 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_reshape_dim_to_batch mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_reshape_dim_to_batch_negative_axis()
{
    const char param_str_ref[] = "7767517\n"
                                 "2 2\n"
                                 "Input   input   0 1 data\n"
                                 "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=3 12=233 13=0\n";

    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=3 12=233 13=-4\n";

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input;
    input.create(W, H, C, B, 4u, 1);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input.channel(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.load_param_mem(param_str_ref);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_negative_axis reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output;
    ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_negative_axis extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_negative_axis value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_output_batch_axis_negative_tail()
{
    const char param_str_ref[] = "7767517\n"
                                 "2 2\n"
                                 "Input   input   0 1 data\n"
                                 "Reshape reshape 1 1 data output 0=2 1=5 11=4 2=3 12=0 13=3\n";

    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=2 1=5 11=4 2=3 12=0 13=-1\n";

    const int B = 2;
    const int C = 3;
    const int H = 4;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.load_param_mem(param_str_ref);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_output_batch_axis_negative_tail reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_output_batch_axis_negative_tail extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_output_batch_axis_negative_tail value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_dim_to_batch_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=3 12=233 13=1\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 2;
    const int C = 3;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, H, C * B, 4u, 1);
    for (int q = 0; q < C; q++)
    {
        for (int b = 0; b < B; b++)
        {
            float* ptr = input.channel(q * B + b);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output_batch;
    int ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis1 extract failed ret=%d\n", ret);
        return -1;
    }
    if (output_batch.n != B || output_batch.dims != 3 || output_batch.w != W || output_batch.h != H || output_batch.c != C)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis1 shape mismatch\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat sub = output_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(b * 100 + q * 20 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_reshape_dim_to_batch_axis1 mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=3 12=1 13=233\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 2;
    const int C = 3;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    int ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1 extract failed ret=%d\n", ret);
        return -1;
    }
    if (output.n != 1 || output.dims != 4 || output.w != W || output.h != H || output.d != B || output.c != C)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1 shape mismatch\n");
        return -1;
    }

    for (int q = 0; q < C; q++)
    {
        const ncnn::Mat sub = output.channel(q);
        for (int b = 0; b < B; b++)
        {
            const float* ptr = sub.channel(b);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(b * 100 + q * 20 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1 mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_axis1_negative_axis()
{
    const char param_str_ref[] = "7767517\n"
                                 "2 2\n"
                                 "Input   input   0 1 data\n"
                                 "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=3 12=1 13=233\n";

    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=3 12=-3 13=233\n";

    const int B = 2;
    const int C = 3;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.load_param_mem(param_str_ref);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1_negative_axis reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1_negative_axis extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1_negative_axis value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_input_batch_axis_negative_tail()
{
    const char param_str_ref[] = "7767517\n"
                                 "2 2\n"
                                 "Input   input   0 1 data\n"
                                 "Reshape reshape 1 1 data output 0=2 1=5 11=4 2=3 12=3 13=233\n";

    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=2 1=5 11=4 2=3 12=-1 13=233\n";

    const int B = 2;
    const int C = 3;
    const int H = 4;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.load_param_mem(param_str_ref);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_input_batch_axis_negative_tail reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_input_batch_axis_negative_tail extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_input_batch_axis_negative_tail value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_axis1_cstep_padding()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=8 1=2 2=3 12=1 13=233\n";

    const int B = 2;
    const int C = 4;
    const int H = 2;
    const int W = 3;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);

    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 1000 + q * 100 + i);
            }
        }
    }

    ncnn::Net net;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    int ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1_cstep_padding extract failed ret=%d\n", ret);
        return -1;
    }

    if (output.dims != 3 || output.w != 8 || output.h != B || output.c != 3)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1_cstep_padding shape mismatch\n");
        return -1;
    }

    for (int q = 0; q < output.c; q++)
    {
        const ncnn::Mat ch = output.channel(q);
        for (int b = 0; b < B; b++)
        {
            const float* ptr = ch.row(b);
            for (int i = 0; i < output.w; i++)
            {
                int index = (q * B + b) * output.w + i;
                int sx = index % W;
                int sy = index / W % H;
                int sb = index / (W * H) % B;
                int sq = index / (W * H * B);
                float expected = (float)(sb * 1000 + sq * 100 + sy * W + sx);
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1_cstep_padding mismatch q=%d b=%d i=%d got %f expect %f\n", q, b, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_reshape_packed_batch_to_dim_axis0()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=-1 12=0 13=233\n";

    const int B = 2;
    const int C = 4;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 1000 + q * 100 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis0 reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_batch_pack4;
    ncnn::convert_packing(input_batch, input_batch_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis0 extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis0 value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_same_axis_reorder()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 6=\"5,3,1,4,-1\" 12=2 13=2\n";

    const int C = 32;
    const int H = 32;
    const int W = 15;

    ncnn::Mat input;
    input.create(W, H, C);
    for (int q = 0; q < C; q++)
    {
        float* ptr = input.channel(q);
        for (int i = 0; i < W * H; i++)
        {
            ptr[i] = (float)(q * 1000 + i);
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_same_axis_reorder reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_pack;
    ncnn::convert_packing(input, input_pack, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_pack);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_same_axis_reorder extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_same_axis_reorder value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_batch_to_dim_axis0_2d()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=8 12=0 13=233\n";

    const int B = 2;
    const int H = 4;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create(W, H, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int y = 0; y < H; y++)
        {
            float* ptr = sub.row(y);
            for (int x = 0; x < W; x++)
            {
                ptr[x] = (float)(b * 1000 + y * 100 + x);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis0_2d reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_batch_pack4;
    ncnn::convert_packing(input_batch, input_batch_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis0_2d extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis0_2d value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=32 12=0 13=233\n";

    const int B = 2;
    const int H = 16;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create(W, H, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int y = 0; y < H; y++)
        {
            float* ptr = sub.row(y);
            for (int x = 0; x < W; x++)
            {
                ptr[x] = (float)(b * 1000 + y * 100 + x);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked_nstep_padding()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=12 12=0 13=233\n";

    const int B = 4;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, (size_t)4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int y = 0; y < H; y++)
        {
            float* ptr = sub.row(y);
            for (int x = 0; x < W; x++)
            {
                ptr[x] = (float)(b * 1000 + y * 100 + x);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked_nstep_padding reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked_nstep_padding extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked_nstep_padding value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_batch_to_dim_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=4 12=1 13=233\n";

    const int B = 2;
    const int C = 4;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 1000 + q * 100 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis1 reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_batch_pack4;
    ncnn::convert_packing(input_batch, input_batch_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis1 extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis1 value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_batch_to_dim_axis1_2d()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=2 2=4 12=1 13=233\n";

    const int B = 2;
    const int H = 4;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create(W, H, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int y = 0; y < H; y++)
        {
            float* ptr = sub.row(y);
            for (int x = 0; x < W; x++)
            {
                ptr[x] = (float)(b * 1000 + y * 100 + x);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis1_2d reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_batch_pack4;
    ncnn::convert_packing(input_batch, input_batch_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis1_2d extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis1_2d value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_axis1_pack1topacked()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=4 12=1 13=233\n";

    const int B = 2;
    const int C = 4;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 1000 + q * 100 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1_pack1topacked reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1_pack1topacked extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1_pack1topacked value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_dim_to_batch_axis0()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=4 2=2 12=233 13=0\n";

    const int B = 2;
    const int C = 4;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, H, C * B, 4u, 1);
    for (int b = 0; b < B; b++)
    {
        for (int q = 0; q < C; q++)
        {
            float* ptr = input.channel(b * C + q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 1000 + q * 100 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis0 reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_pack4;
    ncnn::convert_packing(input, input_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis0 extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis0 value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_dim_to_batch_axis0_2d()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=4 2=2 12=233 13=0\n";

    const int B = 2;
    const int H = 4;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, H * B, (size_t)4u, 1);
    for (int b = 0; b < B; b++)
    {
        for (int y = 0; y < H; y++)
        {
            float* ptr = input.row(b * H + y);
            for (int x = 0; x < W; x++)
            {
                ptr[x] = (float)(b * 1000 + y * 100 + x);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis0_2d reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_pack4;
    ncnn::convert_packing(input, input_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis0_2d extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis0_2d value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_dim_to_batch_axis0_2d_pack1topacked()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=16 2=2 12=233 13=0\n";

    const int B = 2;
    const int H = 16;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, H * B, (size_t)4u, 1);
    for (int b = 0; b < B; b++)
    {
        for (int y = 0; y < H; y++)
        {
            float* ptr = input.row(b * H + y);
            for (int x = 0; x < W; x++)
            {
                ptr[x] = (float)(b * 1000 + y * 100 + x);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis0_2d_pack1topacked reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis0_2d_pack1topacked extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis0_2d_pack1topacked value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_dim_to_batch_axis0_2d_pack4topack1_nstep_padding()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 2=4 12=233 13=0\n";

    const int B = 4;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input;
    input.create(W, H * B, (size_t)4u, 1);
    for (int b = 0; b < B; b++)
    {
        for (int y = 0; y < H; y++)
        {
            float* ptr = input.row(b * H + y);
            for (int x = 0; x < W; x++)
            {
                ptr[x] = (float)(b * 1000 + y * 100 + x);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis0_2d_pack4topack1_nstep_padding reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_pack4;
    ncnn::convert_packing(input, input_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis0_2d_pack4topack1_nstep_padding extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis0_2d_pack4topack1_nstep_padding value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_dim_to_batch_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=4 12=233 13=1\n";

    const int B = 2;
    const int C = 4;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, H, C * B, 4u, 1);
    for (int q = 0; q < C; q++)
    {
        for (int b = 0; b < B; b++)
        {
            float* ptr = input.channel(q * B + b);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 1000 + q * 100 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1 reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_pack4;
    ncnn::convert_packing(input, input_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1 extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1 value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_dim_to_batch_axis1_2d()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=2 2=4 12=233 13=1\n";

    const int B = 2;
    const int H = 4;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, B, H, 4u, 1);
    for (int y = 0; y < H; y++)
    {
        ncnn::Mat sub = input.channel(y);
        for (int b = 0; b < B; b++)
        {
            float* ptr = sub.row(b);
            for (int x = 0; x < W; x++)
            {
                ptr[x] = (float)(b * 1000 + y * 100 + x);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1_2d reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_pack4;
    ncnn::convert_packing(input, input_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1_2d extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1_2d value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_dim_to_batch_axis1_4d()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=4 12=233 13=1\n";

    const int B = 2;
    const int C = 4;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, H, B, C, 4u, 1);
    for (int q = 0; q < C; q++)
    {
        ncnn::Mat sub = input.channel(q);
        for (int b = 0; b < B; b++)
        {
            float* ptr = sub.channel(b);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 1000 + q * 100 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1_4d reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_pack4;
    ncnn::convert_packing(input, input_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1_4d extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1_4d value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_dim_to_batch_axis1_negative_axis()
{
    const char param_str_ref[] = "7767517\n"
                                 "2 2\n"
                                 "Input   input   0 1 data\n"
                                 "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=3 12=233 13=1\n";

    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=3 12=233 13=-3\n";

    const int B = 2;
    const int C = 3;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, H, C * B, 4u, 1);
    for (int q = 0; q < C; q++)
    {
        for (int b = 0; b < B; b++)
        {
            float* ptr = input.channel(q * B + b);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.load_param_mem(param_str_ref);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis1_negative_axis reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output;
    ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis1_negative_axis extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis1_negative_axis value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_dim_to_batch_axis1_pack1topacked()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=4 12=233 13=1\n";

    const int B = 2;
    const int C = 4;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input;
    input.create(W, H, B, C, 4u, 1);
    for (int q = 0; q < C; q++)
    {
        ncnn::Mat sub = input.channel(q);
        for (int b = 0; b < B; b++)
        {
            float* ptr = sub.channel(b);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 1000 + q * 100 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis1_pack1topacked reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis1_pack1topacked extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis1_pack1topacked value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_pack1topacked()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 2=12 12=0 13=233\n";

    const int B = 4;
    const int C = 3;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack1topacked reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack1topacked extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack1topacked value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_pack1tohighpack()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 2=16 12=0 13=233\n";

    const int B = 4;
    const int C = 4;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack1tohighpack reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack1tohighpack extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack1tohighpack value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_batch_to_dim_pack4topack1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=60 1=2 12=0 13=233\n";

    const int B = 2;
    const int C = 4;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack4topack1 reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_batch_pack4;
    ncnn::convert_packing(input_batch, input_batch_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack4topack1 extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack4topack1 value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_dim_to_batch_pack1topacked()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=4 2=3 12=233 13=0\n";

    const int B = 3;
    const int C = 4;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input;
    input.create(W, H, C, B, 4u, 1);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input.channel(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack1topacked reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack1topacked extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack1topacked value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_dim_to_batch_pack1tohighpack()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=16 2=2 12=233 13=0\n";

    const int B = 2;
    const int C = 16;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input;
    input.create(W, H, C * B, 4u, 1);
    for (int b = 0; b < B; b++)
    {
        for (int q = 0; q < C; q++)
        {
            float* ptr = input.channel(b * C + q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack1tohighpack reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack1tohighpack extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack1tohighpack value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_dim_to_batch_pack4topack1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=3 2=4 12=233 13=0\n";

    const int B = 4;
    const int C = 3;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input;
    input.create(W, H, C, B, 4u, 1);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input.channel(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack4topack1 reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;

    ncnn::Mat input_pack4;
    ncnn::convert_packing(input, input_pack4, 4, opt);

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_pack4);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack4topack1 extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack4topack1 value mismatch\n");
        return -1;
    }

    return 0;
}

#if NCNN_BF16
static int test_batch_reshape_bf16_storage_packed()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 2=12 12=0 13=233\n";

    const int B = 4;
    const int C = 3;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 16 + q * 4 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_packed reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.opt.use_fp16_storage = false;
    net.opt.use_bf16_storage = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_packed extract failed ret=%d\n", ret);
        return -1;
    }
    ncnn::Mat output32 = output;
    if (output.elembits() == 16)
        ncnn::cast_bfloat16_to_float32(output, output32, net.opt);
    if (CompareMat(output_ref, output32, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_packed value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_bf16_storage_dim_to_batch_packed()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=12 2=2 12=233 13=0\n";

    const int B = 2;
    const int C = 12;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input;
    input.create(W, H, C * B, 4u, 1);
    for (int b = 0; b < B; b++)
    {
        for (int q = 0; q < C; q++)
        {
            float* ptr = input.channel(b * C + q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 16 + q * 4 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_dim_to_batch_packed reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.opt.use_fp16_storage = false;
    net.opt.use_bf16_storage = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_dim_to_batch_packed extract failed ret=%d\n", ret);
        return -1;
    }
    ncnn::Mat output32 = output;
    if (output.elembits() == 16)
        ncnn::cast_bfloat16_to_float32(output, output32, net.opt);
    if (CompareMat(output_ref, output32, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_dim_to_batch_packed value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_bf16_storage_axis1_packed()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=4 12=1 13=233\n";

    const int B = 2;
    const int C = 4;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 16 + q * 4 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_axis1_packed reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_packing_layout = true;
    net.opt.use_fp16_storage = false;
    net.opt.use_bf16_storage = true;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_axis1_packed extract failed ret=%d\n", ret);
        return -1;
    }
    ncnn::Mat output32 = output;
    if (output.elembits() == 16)
        ncnn::cast_bfloat16_to_float32(output, output32, net.opt);
    if (CompareMat(output_ref, output32, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_axis1_packed value mismatch\n");
        return -1;
    }

    return 0;
}
#endif // NCNN_BF16

static int test_batch_reshape_dim_to_batch_no_infer()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=-1 1=3 11=2 2=-1 12=233 13=0\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    ncnn::Mat input;
    input.create(5, 3, 2, 3, 4u, 1);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output;
    int ret = ex.extract("output", output);
    if (ret == 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_no_infer should fail\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_roundtrip_axis1()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input   input   0 1 data\n"
                             "Reshape d2b     1 1 data tmp 0=7 1=5 11=2 2=3 12=233 13=1\n"
                             "Reshape b2d     1 1 tmp output 0=7 1=5 11=2 2=3 12=1 13=233\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 2;
    const int C = 3;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, H, B, C, 4u, 1);
    for (int q = 0; q < C; q++)
    {
        ncnn::Mat sub = input.channel(q);
        for (int b = 0; b < B; b++)
        {
            float* ptr = sub.channel(b);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output;
    int ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_roundtrip_axis1 extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(input, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_roundtrip_axis1 value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_roundtrip_axis2()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input   input   0 1 data\n"
                             "Reshape d2b     1 1 data tmp 0=7 1=2 11=5 2=3 12=233 13=2\n"
                             "Reshape b2d     1 1 tmp output 0=7 1=2 11=5 2=3 12=2 13=233\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 2;
    const int C = 3;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, B, H, C, 4u, 1);
    for (int q = 0; q < C; q++)
    {
        ncnn::Mat sub = input.channel(q);
        for (int y = 0; y < H; y++)
        {
            float* ptr = sub.channel(y);
            for (int b = 0; b < B; b++)
            {
                for (int x = 0; x < W; x++)
                {
                    ptr[b * W + x] = (float)(b * 100 + q * 20 + y * W + x);
                }
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output;
    int ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_roundtrip_axis2 extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(input, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_roundtrip_axis2 value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_roundtrip()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input   input    0 1 data\n"
                             "Reshape b2d      1 1 data tmp 0=5 1=3 11=2 2=-1 12=0 13=233\n"
                             "Reshape d2b      1 1 tmp output 0=5 1=3 11=2 2=3 12=233 13=0\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output_batch;
    int ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_roundtrip extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(input_batch, output_batch, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_roundtrip value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_permute_fold()
{
    const char param_str[] = "7767517\n"
                             "4 4\n"
                             "Input   input   0 1 data\n"
                             "Reshape b2d     1 1 data tmp0 0=5 1=3 11=2 2=3 12=0 13=233\n"
                             "Permute permute 1 1 tmp0 tmp1 0=6\n"
                             "Reshape reshape 1 1 tmp1 output 0=5 1=3 2=6\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    int ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_permute_fold extract failed ret=%d\n", ret);
        return -1;
    }
    if (output.n != 1 || output.dims != 3 || output.w != W || output.h != H || output.c != B * C)
    {
        fprintf(stderr, "test_batch_reshape_permute_fold shape mismatch\n");
        return -1;
    }

    for (int q = 0; q < C; q++)
    {
        for (int b = 0; b < B; b++)
        {
            const float* ptr = output.channel(q * B + b);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(b * 100 + q * 20 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_reshape_permute_fold mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_reshape_permute_extract()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input   input   0 1 data\n"
                             "Permute permute 1 1 data tmp 0=6\n"
                             "Reshape d2b     1 1 tmp output 0=5 1=3 11=2 2=3 12=233 13=0\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input;
    input.create(W, H, B, C, 4u, 1);
    for (int q = 0; q < C; q++)
    {
        ncnn::Mat sub = input.channel(q);
        for (int b = 0; b < B; b++)
        {
            float* ptr = sub.channel(b);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output_batch;
    int ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_reshape_permute_extract extract failed ret=%d\n", ret);
        return -1;
    }
    if (output_batch.n != B || output_batch.dims != 3 || output_batch.w != W || output_batch.h != H || output_batch.c != C)
    {
        fprintf(stderr, "test_batch_reshape_permute_extract shape mismatch\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat sub = output_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(b * 100 + q * 20 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_reshape_permute_extract mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_fill()
{
    ncnn::Mat m;
    m.create(4, 3, 2, 4u, 1, 3);
    m.fill(7.f);

    for (int b = 0; b < m.n; b++)
    {
        const ncnn::Mat mb = m.batch(b);
        for (int q = 0; q < m.c; q++)
        {
            const float* ptr = mb.channel(q);
            for (int i = 0; i < m.w * m.h; i++)
            {
                if (!NearlyEqual(ptr[i], 7.f, 1e-5f))
                {
                    fprintf(stderr, "test_batch_fill mismatch b=%d q=%d i=%d got %f\n", b, q, i, ptr[i]);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_substract_mean_normalize()
{
    ncnn::Mat m;
    m.create(4, 3, 3, 4u, 1, 2);

    for (int b = 0; b < m.n; b++)
    {
        ncnn::Mat mb = m.batch(b);
        for (int q = 0; q < m.c; q++)
        {
            float* ptr = mb.channel(q);
            for (int i = 0; i < m.w * m.h; i++)
            {
                ptr[i] = (float)(b * 100 + q * 10 + i);
            }
        }
    }

    const float mean_vals[3] = {1.f, 2.f, 3.f};
    const float norm_vals[3] = {0.5f, 1.5f, 2.f};
    m.substract_mean_normalize(mean_vals, norm_vals);

    for (int b = 0; b < m.n; b++)
    {
        const ncnn::Mat mb = m.batch(b);
        for (int q = 0; q < m.c; q++)
        {
            const float* ptr = mb.channel(q);
            for (int i = 0; i < m.w * m.h; i++)
            {
                float v = (float)(b * 100 + q * 10 + i);
                float expected = (v - mean_vals[q]) * norm_vals[q];
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_substract_mean_normalize mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
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
    m.create(8, 6, 3, 4u, 1, 1);

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
    m.create(100, (size_t)4u, 1, 4);

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
    m.create(10, 20, 4u, 1, 3);

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

static int test_batch_forward_binaryop_same_batch()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input    input0 0 1 a\n"
                             "Input    input1 0 1 b\n"
                             "BinaryOp add    2 1 a b out 0=0\n";

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat a;
    ncnn::Mat b;
    a.create(W, H, C, 4u, 1, B);
    b.create(W, H, C, 4u, 1, B);

    for (int bi = 0; bi < B; bi++)
    {
        ncnn::Mat a0 = a.batch(bi);
        ncnn::Mat b0 = b.batch(bi);
        for (int q = 0; q < C; q++)
        {
            float* aptr = a0.channel(q);
            float* bptr = b0.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                aptr[i] = (float)(bi * 100 + q * 10 + i);
                bptr[i] = (float)(bi * 7 + q * 3 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.opt.use_fp16_storage = false;
    net_ref.opt.use_fp16_arithmetic = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("a", a);
    ex_ref.input("b", b);

    ncnn::Mat out_ref;
    int ret = ex_ref.extract("out", out_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_binaryop_same_batch reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("a", a);
    ex.input("b", b);

    ncnn::Mat out;
    ret = ex.extract("out", out);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_binaryop_same_batch extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(out_ref, out, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_forward_binaryop_same_batch value mismatch\n");
        return -1;
    }

#if NCNN_VULKAN
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("a", a);
        ex.input("b", b);

        ncnn::Mat out;
        ret = ex.extract("out", out);
        if (ret != 0)
        {
            fprintf(stderr, "test_batch_forward_binaryop_same_batch vulkan extract failed ret=%d\n", ret);
            return -1;
        }
        if (CompareMat(out_ref, out, 1e-4f) != 0)
        {
            fprintf(stderr, "test_batch_forward_binaryop_same_batch vulkan value mismatch\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

static int test_batch_forward_binaryop_broadcast()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input    input0 0 1 a\n"
                             "Input    input1 0 1 b\n"
                             "BinaryOp add    2 1 a b out 0=0\n";

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat a;
    ncnn::Mat b(W, H, C);
    a.create(W, H, C, 4u, 1, B);

    for (int bi = 0; bi < B; bi++)
    {
        ncnn::Mat a0 = a.batch(bi);
        for (int q = 0; q < C; q++)
        {
            float* ptr = a0.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(bi * 100 + q * 10 + i);
            }
        }
    }
    for (int q = 0; q < C; q++)
    {
        float* ptr = b.channel(q);
        for (int i = 0; i < W * H; i++)
        {
            ptr[i] = (float)(q * 3 + i);
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.opt.use_fp16_storage = false;
    net_ref.opt.use_fp16_arithmetic = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("a", a);
    ex_ref.input("b", b);

    ncnn::Mat out_ref;
    int ret = ex_ref.extract("out", out_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_binaryop_broadcast reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("a", a);
    ex.input("b", b);

    ncnn::Mat out;
    ret = ex.extract("out", out);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_binaryop_broadcast extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(out_ref, out, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_forward_binaryop_broadcast value mismatch\n");
        return -1;
    }

#if NCNN_VULKAN
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("a", a);
        ex.input("b", b);

        ncnn::Mat out;
        ret = ex.extract("out", out);
        if (ret != 0)
        {
            fprintf(stderr, "test_batch_forward_binaryop_broadcast vulkan extract failed ret=%d\n", ret);
            return -1;
        }
        if (CompareMat(out_ref, out, 1e-4f) != 0)
        {
            fprintf(stderr, "test_batch_forward_binaryop_broadcast vulkan value mismatch\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

static int test_batch_forward_scale_external()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input input0 0 1 data\n"
                             "Input input1 0 1 scale\n"
                             "Scale scale0 2 1 data scale out 0=-233\n";

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat data;
    data.create(W, H, C, 4u, 1, B);
    ncnn::Mat scale(C);

    for (int bi = 0; bi < B; bi++)
    {
        ncnn::Mat data0 = data.batch(bi);
        for (int q = 0; q < C; q++)
        {
            float* ptr = data0.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(bi * 100 + q * 10 + i);
            }
        }
    }
    for (int q = 0; q < C; q++)
    {
        scale[q] = (float)(q + 2);
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.opt.use_fp16_storage = false;
    net_ref.opt.use_fp16_arithmetic = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", data);
    ex_ref.input("scale", scale);

    ncnn::Mat out_ref;
    int ret = ex_ref.extract("out", out_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_scale_external reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", data);
    ex.input("scale", scale);

    ncnn::Mat out;
    ret = ex.extract("out", out);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_scale_external extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(out_ref, out, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_forward_scale_external value mismatch\n");
        return -1;
    }

#if NCNN_VULKAN
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", data);
        ex.input("scale", scale);

        ncnn::Mat out;
        ret = ex.extract("out", out);
        if (ret != 0)
        {
            fprintf(stderr, "test_batch_forward_scale_external vulkan extract failed ret=%d\n", ret);
            return -1;
        }
        if (CompareMat(out_ref, out, 1e-4f) != 0)
        {
            fprintf(stderr, "test_batch_forward_scale_external vulkan value mismatch\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

static int test_batch_forward_binaryop_mismatch()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input    input0 0 1 a\n"
                             "Input    input1 0 1 b\n"
                             "BinaryOp add    2 1 a b out 0=0\n";

    ncnn::Net net;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.load_param_mem(param_str);

    ncnn::Mat a;
    ncnn::Mat b;
    a.create(4, 3, 2, 4u, 1, 3);
    b.create(4, 3, 2, 4u, 1, 2);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("a", a);
    ex.input("b", b);

    ncnn::Mat out;
    int ret = ex.extract("out", out);
    if (ret == 0)
    {
        fprintf(stderr, "test_batch_forward_binaryop_mismatch should fail\n");
        return -1;
    }

#if NCNN_VULKAN
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("a", a);
        ex.input("b", b);

        ncnn::Mat out;
        ret = ex.extract("out", out);
        if (ret == 0)
        {
            fprintf(stderr, "test_batch_forward_binaryop_mismatch vulkan should fail\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

static int test_batch_forward_split()
{
    const char param_str[] = "7767517\n"
                             "2 3\n"
                             "Input input 0 1 data\n"
                             "Split split 1 2 data out0 out1\n";

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
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

    ncnn::Net net_ref;
    net_ref.opt.lightmode = false;
    net_ref.opt.use_packing_layout = false;
    net_ref.opt.use_fp16_storage = false;
    net_ref.opt.use_fp16_arithmetic = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat out0_ref;
    ncnn::Mat out1_ref;
    int ret0 = ex_ref.extract("out0", out0_ref);
    int ret1 = ex_ref.extract("out1", out1_ref);
    if (ret0 != 0 || ret1 != 0)
    {
        fprintf(stderr, "test_batch_forward_split reference extract failed ret0=%d ret1=%d\n", ret0, ret1);
        return -1;
    }

    ncnn::Net net;
    net.opt.lightmode = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat out0;
    ncnn::Mat out1;
    ret0 = ex.extract("out0", out0);
    ret1 = ex.extract("out1", out1);
    if (ret0 != 0 || ret1 != 0)
    {
        fprintf(stderr, "test_batch_forward_split extract failed ret0=%d ret1=%d\n", ret0, ret1);
        return -1;
    }
    if (CompareMat(out0_ref, out0, 1e-5f) != 0 || CompareMat(out1_ref, out1, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_forward_split value mismatch\n");
        return -1;
    }

#if NCNN_VULKAN
    {
        ncnn::Net net;
        net.opt.lightmode = false;
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", input_batch);

        ncnn::Mat out0;
        ncnn::Mat out1;
        ret0 = ex.extract("out0", out0);
        ret1 = ex.extract("out1", out1);
        if (ret0 != 0 || ret1 != 0)
        {
            fprintf(stderr, "test_batch_forward_split vulkan extract failed ret0=%d ret1=%d\n", ret0, ret1);
            return -1;
        }
        if (CompareMat(out0_ref, out0, 1e-4f) != 0 || CompareMat(out1_ref, out1, 1e-4f) != 0)
        {
            fprintf(stderr, "test_batch_forward_split vulkan value mismatch\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

static int test_batch_forward_flatten()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Flatten flatten 1 1 data output\n";

    const int B = 2;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
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

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.opt.use_fp16_storage = false;
    net_ref.opt.use_fp16_arithmetic = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_flatten reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output_batch;
    ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_flatten extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output_batch, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_forward_flatten value mismatch\n");
        return -1;
    }

#if NCNN_VULKAN
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", input_batch);

        ncnn::Mat output_batch;
        ret = ex.extract("output", output_batch);
        if (ret != 0)
        {
            fprintf(stderr, "test_batch_forward_flatten vulkan extract failed ret=%d\n", ret);
            return -1;
        }
        if (CompareMat(output_ref, output_batch, 1e-4f) != 0)
        {
            fprintf(stderr, "test_batch_forward_flatten vulkan value mismatch\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

static int test_batch_forward_shape_ops()
{
    const char param_str[] = "7767517\n"
                             "5 5\n"
                             "Input      input   0 1 data\n"
                             "Reshape    reshape 1 1 data reshaped 0=6 1=4 2=1\n"
                             "ExpandDims expand  1 1 reshaped expanded -23303=1,1\n"
                             "Squeeze    squeeze 1 1 expanded squeezed -23303=1,1\n"
                             "Flatten    flatten 1 1 squeezed output\n";

    const int B = 2;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
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

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.opt.use_fp16_storage = false;
    net_ref.opt.use_fp16_arithmetic = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_shape_ops reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output_batch;
    ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_shape_ops extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output_batch, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_forward_shape_ops value mismatch\n");
        return -1;
    }

#if NCNN_VULKAN
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", input_batch);

        ncnn::Mat output_batch;
        ret = ex.extract("output", output_batch);
        if (ret != 0)
        {
            fprintf(stderr, "test_batch_forward_shape_ops vulkan extract failed ret=%d\n", ret);
            return -1;
        }
        if (CompareMat(output_ref, output_batch, 1e-4f) != 0)
        {
            fprintf(stderr, "test_batch_forward_shape_ops vulkan value mismatch\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

static int test_batch_forward_relu()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input input 0 1 data\n"
                             "ReLU  relu  1 1 data output 0=1.000000e-01\n";

    const int B = 4;
    const int C = 3;
    const int H = 3;
    const int W = 4;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    if (input_batch.empty())
    {
        fprintf(stderr, "test_batch_forward_relu create failed\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        sub.fill((float)(b - 1.5f));
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.opt.use_fp16_storage = false;
    net_ref.opt.use_fp16_arithmetic = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_relu reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output_batch;
    ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_relu extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output_batch, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_forward_relu value mismatch\n");
        return -1;
    }

#if NCNN_VULKAN
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", input_batch);

        ncnn::Mat output_batch;
        ret = ex.extract("output", output_batch);
        if (ret != 0)
        {
            fprintf(stderr, "test_batch_forward_relu vulkan extract failed ret=%d\n", ret);
            return -1;
        }
        if (CompareMat(output_ref, output_batch, 1e-4f) != 0)
        {
            fprintf(stderr, "test_batch_forward_relu vulkan value mismatch\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

static int test_batch_forward_pooling()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Pooling pooling 1 1 data output 0=0 1=2 2=2\n";

    const int B = 2;
    const int C = 2;
    const int H = 4;
    const int W = 4;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);

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

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.opt.use_fp16_storage = false;
    net_ref.opt.use_fp16_arithmetic = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_pooling reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.load_param_mem(param_str);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output_batch;
    ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_pooling extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output_batch, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_forward_pooling value mismatch\n");
        return -1;
    }

#if NCNN_VULKAN
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", input_batch);

        ncnn::Mat output_batch;
        ret = ex.extract("output", output_batch);
        if (ret != 0)
        {
            fprintf(stderr, "test_batch_forward_pooling vulkan extract failed ret=%d\n", ret);
            return -1;
        }
        if (CompareMat(output_ref, output_batch, 1e-4f) != 0)
        {
            fprintf(stderr, "test_batch_forward_pooling vulkan value mismatch\n");
            return -1;
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

#if NCNN_VULKAN
static int test_vkmat_create_batch_basic()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();

    ncnn::VkMat m;
    m.create(8, 6, 3, 4u, 1, 4, blob_allocator);

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
    m.create(7, 5, 13, 4u, 1, 4, blob_allocator);

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
    m.create(4, 3, 2, 4u, 1, 3, blob_allocator);

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
    m.create(4, 3, 2, 4u, 1, 4, blob_allocator);

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
    m.create(4, 3, 2, 4u, 1, 4, blob_allocator);
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

static int test_vkmat_create_reset()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();

    ncnn::VkMat m;
    m.create(4, 3, 2, 4u, 1, 3, blob_allocator);
    m.create(4, 3, 2, (size_t)4u, blob_allocator);

    if (m.n != 1)
    {
        fprintf(stderr, "test_vkmat_create_reset n expect 1 got %d\n", m.n);
        m.release();
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }
    if (m.dims != 3 || m.w != 4 || m.h != 3 || m.c != 2)
    {
        fprintf(stderr, "test_vkmat_create_reset shape mismatch\n");
        m.release();
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    m.release();
    vkdev->reclaim_blob_allocator(blob_allocator);
    return 0;
}

static int test_vkimage_batch_not_supported()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();

    ncnn::Mat cpu_batch;
    cpu_batch.create(4, 3, 2, 4u, 1, 3);

    ncnn::VkImageMat im;
    im.create_like(cpu_batch, blob_allocator);
    if (!im.empty())
    {
        fprintf(stderr, "test_vkimage_batch_not_supported cpu batch should fail\n");
        im.release();
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    ncnn::VkMat gpu_batch;
    gpu_batch.create(4, 3, 2, 4u, 1, 3, blob_allocator);
    im.create_like(gpu_batch, blob_allocator);
    if (!im.empty())
    {
        fprintf(stderr, "test_vkimage_batch_not_supported gpu batch should fail\n");
        im.release();
        gpu_batch.release();
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    gpu_batch.release();
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
    cpu_batch.create(W, H, C, 4u, 1, B);
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
            gpu_batch.create_like(gpu_b, B, blob_allocator);
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
        if (CompareMat(cpu_batch.batch(b), result, 1e-5) != 0)
        {
            fprintf(stderr, "test_vkmat_batch_upload_download value mismatch at batch %d\n", b);
            vkdev->reclaim_staging_allocator(staging_allocator);
            vkdev->reclaim_blob_allocator(blob_allocator);
            return -1;
        }
    }

    vkdev->reclaim_staging_allocator(staging_allocator);
    vkdev->reclaim_blob_allocator(blob_allocator);
    return 0;
}

static int test_vkmat_batch_upload_download_whole()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    const int B = 3;
    const int W = 4;
    const int H = 3;
    const int C = 2;

    ncnn::Mat cpu_batch;
    cpu_batch.create(W, H, C, 4u, 1, B);
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

    ncnn::VkCompute cmd(vkdev);

    ncnn::Option opt;
    opt.blob_vkallocator = blob_allocator;
    opt.workspace_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;
    opt.use_vulkan_compute = true;

    ncnn::VkMat gpu_batch;
    ncnn::Mat cpu_result;
    cmd.record_upload(cpu_batch, gpu_batch, opt);
    cmd.record_download(gpu_batch, cpu_result, opt);

    int ret = cmd.submit_and_wait();
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_upload_download_whole submit failed ret=%d\n", ret);
        vkdev->reclaim_staging_allocator(staging_allocator);
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    if (cpu_result.n != B || cpu_result.w != W || cpu_result.h != H || cpu_result.c != C)
    {
        fprintf(stderr, "test_vkmat_batch_upload_download_whole shape mismatch\n");
        vkdev->reclaim_staging_allocator(staging_allocator);
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    if (CompareMat(cpu_batch, cpu_result, 1e-5) != 0)
    {
        fprintf(stderr, "test_vkmat_batch_upload_download_whole value mismatch\n");
        vkdev->reclaim_staging_allocator(staging_allocator);
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    vkdev->reclaim_staging_allocator(staging_allocator);
    vkdev->reclaim_blob_allocator(blob_allocator);
    return 0;
}

static int test_vktransfer_batch_upload()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    const int B = 3;
    const int W = 4;
    const int H = 3;
    const int C = 2;

    ncnn::Mat cpu_batch;
    cpu_batch.create(W, H, C, 4u, 1, B);
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

    ncnn::Option opt;
    opt.blob_vkallocator = blob_allocator;
    opt.workspace_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;
    opt.use_vulkan_compute = true;

    ncnn::VkMat gpu_batch;
    {
        ncnn::VkTransfer cmd(vkdev);
        cmd.record_upload(cpu_batch, gpu_batch, opt, false);
        int ret = cmd.submit_and_wait();
        if (ret != 0)
        {
            fprintf(stderr, "test_vktransfer_batch_upload upload submit failed ret=%d\n", ret);
            vkdev->reclaim_staging_allocator(staging_allocator);
            vkdev->reclaim_blob_allocator(blob_allocator);
            return -1;
        }
    }

    ncnn::Mat cpu_result;
    {
        ncnn::VkCompute cmd(vkdev);
        cmd.record_download(gpu_batch, cpu_result, opt);
        int ret = cmd.submit_and_wait();
        if (ret != 0)
        {
            fprintf(stderr, "test_vktransfer_batch_upload download submit failed ret=%d\n", ret);
            vkdev->reclaim_staging_allocator(staging_allocator);
            vkdev->reclaim_blob_allocator(blob_allocator);
            return -1;
        }
    }

    if (cpu_result.n != B || cpu_result.w != W || cpu_result.h != H || cpu_result.c != C)
    {
        fprintf(stderr, "test_vktransfer_batch_upload shape mismatch\n");
        vkdev->reclaim_staging_allocator(staging_allocator);
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    if (CompareMat(cpu_batch, cpu_result, 1e-5) != 0)
    {
        fprintf(stderr, "test_vktransfer_batch_upload value mismatch\n");
        vkdev->reclaim_staging_allocator(staging_allocator);
        vkdev->reclaim_blob_allocator(blob_allocator);
        return -1;
    }

    vkdev->reclaim_staging_allocator(staging_allocator);
    vkdev->reclaim_blob_allocator(blob_allocator);
    return 0;
}

static int test_vkmat_batch_forward_reshape_batch_to_dim()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=-1 12=0 13=233\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    int ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_batch_to_dim extract failed ret=%d\n", ret);
        return -1;
    }
    if (output.n != 1 || output.dims != 4 || output.w != W || output.h != H || output.d != C || output.c != B)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_batch_to_dim shape mismatch\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat sub = output.channel(b);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(b * 100 + q * 20 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-4f))
                {
                    fprintf(stderr, "test_vkmat_batch_forward_reshape_batch_to_dim mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_reshape_dim_to_batch()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=3 12=233 13=0\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input;
    input.create(W, H, C, B, 4u, 1);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input.channel(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output_batch;
    int ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_dim_to_batch extract failed ret=%d\n", ret);
        return -1;
    }
    if (output_batch.n != B || output_batch.dims != 3 || output_batch.w != W || output_batch.h != H || output_batch.c != C)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_dim_to_batch shape mismatch\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat sub = output_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(b * 100 + q * 20 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-4f))
                {
                    fprintf(stderr, "test_vkmat_batch_forward_reshape_dim_to_batch mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_reshape_negative_axis()
{
    const char param_str_ref[] = "7767517\n"
                                 "2 2\n"
                                 "Input   input   0 1 data\n"
                                 "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=-1 12=0 13=233\n";

    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=-1 12=-4 13=233\n";

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str_ref);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_negative_axis reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_negative_axis extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-4f) != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_negative_axis value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_vkmat_batch_forward_reshape_shape_expr()
{
    const char param_str_ref[] = "7767517\n"
                                 "2 2\n"
                                 "Input   input   0 1 data\n"
                                 "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=3 12=0 13=233\n";

    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 6=\"0w,0h,0c,0n\" 12=0 13=233\n";

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str_ref);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_shape_expr reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_shape_expr extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-4f) != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_shape_expr value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_vkmat_batch_forward_reshape_same_axis_relu()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data reshaped 0=4 1=2 11=3 2=3 12=0 13=0\n"
                             "ReLU    relu    1 1 reshaped output 0=1.000000e-01\n";

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i - 8);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_same_axis_relu reference extract failed ret=%d\n", ret);
        return -1;
    }

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_same_axis_relu extract failed ret=%d\n", ret);
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-4f) != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_same_axis_relu value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_vkmat_batch_forward_reshape_dim_to_batch_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=3 12=233 13=1\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    const int B = 2;
    const int C = 3;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, H, C * B, 4u, 1);
    for (int q = 0; q < C; q++)
    {
        for (int b = 0; b < B; b++)
        {
            float* ptr = input.channel(q * B + b);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input);

    ncnn::Mat output_batch;
    int ret = ex.extract("output", output_batch);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_dim_to_batch_axis1 extract failed ret=%d\n", ret);
        return -1;
    }
    if (output_batch.n != B || output_batch.dims != 3 || output_batch.w != W || output_batch.h != H || output_batch.c != C)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_dim_to_batch_axis1 shape mismatch\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat sub = output_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(b * 100 + q * 20 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-4f))
                {
                    fprintf(stderr, "test_vkmat_batch_forward_reshape_dim_to_batch_axis1 mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_reshape_batch_to_dim_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=3 12=1 13=233\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    const int B = 2;
    const int C = 3;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", input_batch);

    ncnn::Mat output;
    int ret = ex.extract("output", output);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_batch_to_dim_axis1 extract failed ret=%d\n", ret);
        return -1;
    }
    if (output.n != 1 || output.dims != 4 || output.w != W || output.h != H || output.d != B || output.c != C)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_batch_to_dim_axis1 shape mismatch\n");
        return -1;
    }

    for (int q = 0; q < C; q++)
    {
        const ncnn::Mat sub = output.channel(q);
        for (int b = 0; b < B; b++)
        {
            const float* ptr = sub.channel(b);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(b * 100 + q * 20 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-4f))
                {
                    fprintf(stderr, "test_vkmat_batch_forward_reshape_batch_to_dim_axis1 mismatch b=%d q=%d i=%d got %f expect %f\n", b, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_reshape_batch_to_dim_pack1to4()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=-1 12=0 13=233\n";

    const int B = 2;
    const int C = 3;
    const int H = 5;
    const int W = 4;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_batch_to_dim_pack1to4 reference extract failed ret=%d\n", ret);
        return -1;
    }

    for (int use_fp16_storage = 0; use_fp16_storage < 2; use_fp16_storage++)
    {
        if (use_fp16_storage && !ncnn::get_gpu_device()->info.support_fp16_storage())
            continue;

        ncnn::Net net;
        ncnn::Option opt;
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = use_fp16_storage;
        opt.use_fp16_storage = use_fp16_storage;
        net.opt = opt;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", input_batch);

        ncnn::Mat output;
        ret = ex.extract("output", output, 1);
        if (ret != 0)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_batch_to_dim_pack1to4 extract failed fp16s=%d ret=%d\n", use_fp16_storage, ret);
            return -1;
        }
        if (CompareMat(output_ref, output, 1e-4f) != 0)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_batch_to_dim_pack1to4 value mismatch fp16s=%d\n", use_fp16_storage);
            return -1;
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_reshape_packed_batch_to_dim_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=4 12=1 13=233\n";

    const int B = 2;
    const int C = 4;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create(W, H, C, 4u, 1, B);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input_batch.batch(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input_batch);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_packed_batch_to_dim_axis1 reference extract failed ret=%d\n", ret);
        return -1;
    }

    for (int use_fp16_storage = 0; use_fp16_storage < 2; use_fp16_storage++)
    {
        if (use_fp16_storage && !ncnn::get_gpu_device()->info.support_fp16_storage())
            continue;

        ncnn::Net net;
        ncnn::Option opt;
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = use_fp16_storage;
        opt.use_fp16_storage = use_fp16_storage;
        net.opt = opt;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", input_batch);

        ncnn::Mat output;
        ret = ex.extract("output", output, 1);
        if (ret != 0)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_packed_batch_to_dim_axis1 extract failed fp16s=%d ret=%d\n", use_fp16_storage, ret);
            return -1;
        }
        if (CompareMat(output_ref, output, 1e-4f) != 0)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_packed_batch_to_dim_axis1 value mismatch fp16s=%d\n", use_fp16_storage);
            return -1;
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_reshape_dim_to_batch_pack4to1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 11=3 2=4 12=233 13=0\n";

    const int B = 4;
    const int C = 3;
    const int H = 3;
    const int W = 5;

    ncnn::Mat input;
    input.create(W, H, C, B, 4u, 1);
    for (int b = 0; b < B; b++)
    {
        ncnn::Mat sub = input.channel(b);
        for (int q = 0; q < C; q++)
        {
            float* ptr = sub.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_dim_to_batch_pack4to1 reference extract failed ret=%d\n", ret);
        return -1;
    }

    for (int use_fp16_storage = 0; use_fp16_storage < 2; use_fp16_storage++)
    {
        if (use_fp16_storage && !ncnn::get_gpu_device()->info.support_fp16_storage())
            continue;

        ncnn::Net net;
        ncnn::Option opt;
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = use_fp16_storage;
        opt.use_fp16_storage = use_fp16_storage;
        net.opt = opt;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", input);

        ncnn::Mat output;
        ret = ex.extract("output", output, 1);
        if (ret != 0)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_dim_to_batch_pack4to1 extract failed fp16s=%d ret=%d\n", use_fp16_storage, ret);
            return -1;
        }
        if (CompareMat(output_ref, output, 1e-4f) != 0)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_dim_to_batch_pack4to1 value mismatch fp16s=%d\n", use_fp16_storage);
            return -1;
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_reshape_packed_dim_to_batch_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=4 12=233 13=1\n";

    const int B = 2;
    const int C = 4;
    const int H = 5;
    const int W = 7;

    ncnn::Mat input;
    input.create(W, H, C * B, 4u, 1);
    for (int q = 0; q < C; q++)
    {
        for (int b = 0; b < B; b++)
        {
            float* ptr = input.channel(q * B + b);
            for (int i = 0; i < W * H; i++)
            {
                ptr[i] = (float)(b * 100 + q * 20 + i);
            }
        }
    }

    ncnn::Net net_ref;
    net_ref.opt.use_packing_layout = false;
    net_ref.load_param_mem(param_str);

    ncnn::Extractor ex_ref = net_ref.create_extractor();
    ex_ref.input("data", input);

    ncnn::Mat output_ref;
    int ret = ex_ref.extract("output", output_ref);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_reshape_packed_dim_to_batch_axis1 reference extract failed ret=%d\n", ret);
        return -1;
    }

    for (int use_fp16_storage = 0; use_fp16_storage < 2; use_fp16_storage++)
    {
        if (use_fp16_storage && !ncnn::get_gpu_device()->info.support_fp16_storage())
            continue;

        ncnn::Net net;
        ncnn::Option opt;
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = use_fp16_storage;
        opt.use_fp16_storage = use_fp16_storage;
        net.opt = opt;
        net.load_param_mem(param_str);
        net.load_model((const unsigned char*)"");

        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", input);

        ncnn::Mat output;
        ret = ex.extract("output", output, 1);
        if (ret != 0)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_packed_dim_to_batch_axis1 extract failed fp16s=%d ret=%d\n", use_fp16_storage, ret);
            return -1;
        }
        if (CompareMat(output_ref, output, 1e-4f) != 0)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_packed_dim_to_batch_axis1 value mismatch fp16s=%d\n", use_fp16_storage);
            return -1;
        }
    }

    return 0;
}

#endif // NCNN_VULKAN

static int test_mat_batch_cpu()
{
    return 0
           || test_create_batch_basic()
           || test_nstep_alignment()
           || test_batch_subview_zero_copy()
           || test_batch_range()
           || test_batch_data_isolation()
           || test_batch_clone()
           || test_batch_release()
           || test_batch_create_reset()
           || test_batch_reshape()
           || test_batch_reshape_zero_copy()
           || test_batch_reshape_batch_to_dim_flatten()
           || test_batch_reshape_batch_to_dim_4d()
           || test_batch_reshape_batch_to_dim_negative_axis()
           || test_batch_reshape_batch_to_dim_shape_expr()
           || test_batch_reshape_dim_to_batch()
           || test_batch_reshape_dim_to_batch_negative_axis()
           || test_batch_reshape_output_batch_axis_negative_tail()
           || test_batch_reshape_dim_to_batch_axis1()
           || test_batch_reshape_batch_to_dim_axis1()
           || test_batch_reshape_batch_to_dim_axis1_negative_axis()
           || test_batch_reshape_input_batch_axis_negative_tail()
           || test_batch_reshape_packed_batch_to_dim_axis0()
           || test_batch_reshape_packed_same_axis_reorder()
           || test_batch_reshape_packed_batch_to_dim_axis0_2d()
           || test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked()
           || test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked_nstep_padding()
           || test_batch_reshape_packed_batch_to_dim_axis1()
           || test_batch_reshape_packed_batch_to_dim_axis1_2d()
           || test_batch_reshape_batch_to_dim_axis1_pack1topacked()
           || test_batch_reshape_batch_to_dim_axis1_cstep_padding()
           || test_batch_reshape_packed_dim_to_batch_axis0()
           || test_batch_reshape_packed_dim_to_batch_axis0_2d()
           || test_batch_reshape_dim_to_batch_axis0_2d_pack1topacked()
           || test_batch_reshape_dim_to_batch_axis0_2d_pack4topack1_nstep_padding()
           || test_batch_reshape_packed_dim_to_batch_axis1()
           || test_batch_reshape_packed_dim_to_batch_axis1_2d()
           || test_batch_reshape_packed_dim_to_batch_axis1_4d()
           || test_batch_reshape_dim_to_batch_axis1_negative_axis()
           || test_batch_reshape_dim_to_batch_axis1_pack1topacked()
           || test_batch_reshape_batch_to_dim_pack1topacked()
           || test_batch_reshape_batch_to_dim_pack1tohighpack()
           || test_batch_reshape_batch_to_dim_pack4topack1()
           || test_batch_reshape_dim_to_batch_pack1topacked()
           || test_batch_reshape_dim_to_batch_pack1tohighpack()
           || test_batch_reshape_dim_to_batch_pack4topack1()
#if NCNN_BF16
           || test_batch_reshape_bf16_storage_packed()
           || test_batch_reshape_bf16_storage_dim_to_batch_packed()
           || test_batch_reshape_bf16_storage_axis1_packed()
#endif // NCNN_BF16
           || test_batch_reshape_dim_to_batch_no_infer()
           || test_batch_reshape_roundtrip_axis1()
           || test_batch_reshape_roundtrip_axis2()
           || test_batch_reshape_roundtrip()
           || test_batch_reshape_permute_fold()
           || test_batch_reshape_permute_extract()
           || test_batch_fill()
           || test_batch_substract_mean_normalize()
           || test_backward_compatibility()
           || test_create_batch_single()
           || test_create_batch_1d()
           || test_create_batch_2d();
}

static int test_batch_forward()
{
    return 0
           || test_batch_forward_binaryop_same_batch()
           || test_batch_forward_binaryop_broadcast()
           || test_batch_forward_scale_external()
           || test_batch_forward_binaryop_mismatch()
           || test_batch_forward_split()
           || test_batch_forward_flatten()
           || test_batch_forward_shape_ops()
           || test_batch_forward_relu()
           || test_batch_forward_pooling();
}

#if NCNN_VULKAN
static int test_vkmat_batch()
{
    return 0
           || test_vkmat_create_batch_basic()
           || test_vkmat_nstep_alignment()
           || test_vkmat_batch_subview()
           || test_vkmat_batch_range()
           || test_vkmat_batch_release()
           || test_vkmat_create_reset()
           || test_vkimage_batch_not_supported()
           || test_vkmat_batch_upload_download()
           || test_vkmat_batch_upload_download_whole()
           || test_vktransfer_batch_upload()
           || test_vkmat_batch_forward_reshape_batch_to_dim()
           || test_vkmat_batch_forward_reshape_dim_to_batch()
           || test_vkmat_batch_forward_reshape_negative_axis()
           || test_vkmat_batch_forward_reshape_shape_expr()
           || test_vkmat_batch_forward_reshape_same_axis_relu()
           || test_vkmat_batch_forward_reshape_dim_to_batch_axis1()
           || test_vkmat_batch_forward_reshape_batch_to_dim_axis1()
           || test_vkmat_batch_forward_reshape_batch_to_dim_pack1to4()
           || test_vkmat_batch_forward_reshape_packed_batch_to_dim_axis1()
           || test_vkmat_batch_forward_reshape_dim_to_batch_pack4to1()
           || test_vkmat_batch_forward_reshape_packed_dim_to_batch_axis1();
}
#endif // NCNN_VULKAN

int main()
{
    int ret = test_mat_batch_cpu();
    if (ret != 0)
        return ret;

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN

    ret = test_batch_forward();
    if (ret != 0)
    {
#if NCNN_VULKAN
        ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN
        return ret;
    }

#if NCNN_VULKAN
    if (ncnn::get_gpu_count() > 0)
    {
        ret = test_vkmat_batch();
    }
    else
    {
        fprintf(stderr, "no vulkan device, skip vkmat batch tests\n");
    }
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    return ret;
}
