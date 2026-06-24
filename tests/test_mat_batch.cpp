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

static int test_batch_create_reset()
{
    ncnn::Mat m;
    m.create_batch(4, 3, 2, 3, 4u, 1);
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
    m.create_batch(W, H, C, B, 4u, 1);

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
    m.create_batch(W, H, C, B, 4u, 1);

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
                             "Reshape reshape 1 1 data output 0=-1 12=1\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

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
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=-1 12=1\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

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

static int test_batch_reshape_dim_to_batch()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 2=2 12=2\n";

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

static int test_batch_reshape_dim_to_batch_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 2=3 12=2 13=1\n";

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
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=3 12=1 13=1\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 2;
    const int C = 3;
    const int H = 5;
    const int W = 7;

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

static int test_batch_reshape_packed_batch_to_dim_axis0()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=-1 12=1\n";

    const int B = 2;
    const int C = 4;
    const int H = 5;
    const int W = 7;

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis0 output should stay packed\n");
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis0 value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_batch_to_dim_axis0_2d()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=8 12=1\n";

    const int B = 2;
    const int H = 4;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create_batch(W, H, B, 4u, 1);
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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis0_2d output should stay packed\n");
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
                             "Reshape reshape 1 1 data output 0=7 1=32 12=1\n";

    const int B = 2;
    const int H = 16;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create_batch(W, H, B, 4u, 1);
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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked output should be packed\n");
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_batch_to_dim_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=4 12=1 13=1\n";

    const int B = 2;
    const int C = 4;
    const int H = 5;
    const int W = 7;

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis1 output should stay packed\n");
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
                             "Reshape reshape 1 1 data output 0=7 1=2 2=4 12=1 13=1\n";

    const int B = 2;
    const int H = 4;
    const int W = 7;

    ncnn::Mat input_batch;
    input_batch.create_batch(W, H, B, 4u, 1);
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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_packed_batch_to_dim_axis1_2d output should stay packed\n");
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
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=4 12=1 13=1\n";

    const int B = 2;
    const int C = 4;
    const int H = 3;
    const int W = 5;

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_axis1_pack1topacked output should be packed\n");
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
                             "Reshape reshape 1 1 data output 0=7 1=5 2=4 12=2\n";

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis0 output should stay packed\n");
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
                             "Reshape reshape 1 1 data output 0=7 1=4 12=2\n";

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis0_2d output should stay packed\n");
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
                             "Reshape reshape 1 1 data output 0=7 1=16 12=2\n";

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis0_2d_pack1topacked output should be packed\n");
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis0_2d_pack1topacked value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_packed_dim_to_batch_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 2=4 12=2 13=1\n";

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1 output should stay packed\n");
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
                             "Reshape reshape 1 1 data output 0=7 1=4 12=2 13=1\n";

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1_2d output should stay packed\n");
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
                             "Reshape reshape 1 1 data output 0=7 1=5 2=4 12=2 13=1\n";

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1_4d output should stay packed\n");
        return -1;
    }
    if (CompareMat(output_ref, output, 1e-5f) != 0)
    {
        fprintf(stderr, "test_batch_reshape_packed_dim_to_batch_axis1_4d value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_reshape_dim_to_batch_axis1_pack1topacked()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=5 1=3 2=4 12=2 13=1\n";

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_axis1_pack1topacked output should be packed\n");
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
                             "Reshape reshape 1 1 data output 0=5 1=3 2=12 12=1\n";

    const int B = 4;
    const int C = 3;
    const int H = 3;
    const int W = 5;

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack1topacked output should be packed\n");
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
                             "Reshape reshape 1 1 data output 0=5 1=3 2=16 12=1\n";

    const int B = 4;
    const int C = 4;
    const int H = 3;
    const int W = 5;

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack1tohighpack output should be packed\n");
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
                             "Reshape reshape 1 1 data output 0=60 1=2 12=1\n";

    const int B = 2;
    const int C = 4;
    const int H = 3;
    const int W = 5;

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
    if (output.elempack != 1)
    {
        fprintf(stderr, "test_batch_reshape_batch_to_dim_pack4topack1 output should be pack1\n");
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
                             "Reshape reshape 1 1 data output 0=5 1=3 2=4 12=2\n";

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack1topacked output should be packed\n");
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
                             "Reshape reshape 1 1 data output 0=5 1=3 2=16 12=2\n";

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack1tohighpack output should be packed\n");
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
                             "Reshape reshape 1 1 data output 0=5 1=3 2=3 12=2\n";

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
    if (output.elempack != 1)
    {
        fprintf(stderr, "test_batch_reshape_dim_to_batch_pack4topack1 output should be pack1\n");
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
                             "Reshape reshape 1 1 data output 0=5 1=3 2=12 12=1\n";

    const int B = 4;
    const int C = 3;
    const int H = 3;
    const int W = 5;

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_packed output should be packed\n");
        return -1;
    }
    if (output.elembits() != 16)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_packed output should be bf16 storage\n");
        return -1;
    }

    ncnn::Mat output32;
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
                             "Reshape reshape 1 1 data output 0=5 1=3 2=12 12=2\n";

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_dim_to_batch_packed output should be packed\n");
        return -1;
    }
    if (output.elembits() != 16)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_dim_to_batch_packed output should be bf16 storage\n");
        return -1;
    }

    ncnn::Mat output32;
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
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=4 12=1 13=1\n";

    const int B = 2;
    const int C = 4;
    const int H = 3;
    const int W = 5;

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
    if (output.elempack == 1)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_axis1_packed output should be packed\n");
        return -1;
    }
    if (output.elembits() != 16)
    {
        fprintf(stderr, "test_batch_reshape_bf16_storage_axis1_packed output should be bf16 storage\n");
        return -1;
    }

    ncnn::Mat output32;
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
                             "Reshape reshape 1 1 data output 0=-1 1=3 2=2 12=2\n";

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
                             "Reshape d2b     1 1 data tmp 0=7 1=5 2=3 12=2 13=1\n"
                             "Reshape b2d     1 1 tmp output 0=7 1=5 11=2 2=3 12=1 13=1\n";

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
                             "Reshape d2b     1 1 data tmp 0=7 1=5 2=3 12=2 13=2\n"
                             "Reshape b2d     1 1 tmp output 0=7 1=2 11=5 2=3 12=1 13=2\n";

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
                             "Reshape b2d      1 1 data tmp 0=5 1=3 11=2 2=-1 12=1\n"
                             "Reshape d2b      1 1 tmp output 0=5 1=3 2=2 12=2\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

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
                             "Reshape b2d     1 1 data tmp0 0=5 1=3 11=2 2=3 12=1\n"
                             "Permute permute 1 1 tmp0 tmp1 0=6\n"
                             "Reshape reshape 1 1 tmp1 output 0=5 1=3 2=6\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

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
                             "Reshape d2b     1 1 tmp output 0=5 1=3 2=2 12=2\n";

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

static int test_batch_forward_binaryop_same_batch()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input    input0 0 1 a\n"
                             "Input    input1 0 1 b\n"
                             "BinaryOp add    2 1 a b out 0=0\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat a;
    ncnn::Mat b;
    a.create_batch(W, H, C, B, 4u, 1);
    b.create_batch(W, H, C, B, 4u, 1);

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

    ncnn::Extractor ex = net.create_extractor();
    ex.input("a", a);
    ex.input("b", b);

    ncnn::Mat out;
    int ret = ex.extract("out", out);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_binaryop_same_batch extract failed ret=%d\n", ret);
        return -1;
    }
    if (out.n != B || out.w != W || out.h != H || out.c != C)
    {
        fprintf(stderr, "test_batch_forward_binaryop_same_batch shape mismatch\n");
        return -1;
    }

    for (int bi = 0; bi < B; bi++)
    {
        const ncnn::Mat out0 = out.batch(bi);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = out0.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(bi * 100 + q * 10 + i + bi * 7 + q * 3 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_forward_binaryop_same_batch mismatch at b=%d q=%d i=%d got %f expect %f\n", bi, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_forward_binaryop_broadcast()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input    input0 0 1 a\n"
                             "Input    input1 0 1 b\n"
                             "BinaryOp add    2 1 a b out 0=0\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat a;
    ncnn::Mat b(W, H, C);
    a.create_batch(W, H, C, B, 4u, 1);

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

    ncnn::Extractor ex = net.create_extractor();
    ex.input("a", a);
    ex.input("b", b);

    ncnn::Mat out;
    int ret = ex.extract("out", out);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_binaryop_broadcast extract failed ret=%d\n", ret);
        return -1;
    }
    if (out.n != B || out.w != W || out.h != H || out.c != C)
    {
        fprintf(stderr, "test_batch_forward_binaryop_broadcast shape mismatch\n");
        return -1;
    }

    for (int bi = 0; bi < B; bi++)
    {
        const ncnn::Mat out0 = out.batch(bi);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = out0.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(bi * 100 + q * 10 + i + q * 3 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_forward_binaryop_broadcast mismatch at b=%d q=%d i=%d got %f expect %f\n", bi, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_batch_forward_scale_external()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input input0 0 1 data\n"
                             "Input input1 0 1 scale\n"
                             "Scale scale0 2 1 data scale out 0=-233\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat data;
    data.create_batch(W, H, C, B, 4u, 1);
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

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", data);
    ex.input("scale", scale);

    ncnn::Mat out;
    int ret = ex.extract("out", out);
    if (ret != 0)
    {
        fprintf(stderr, "test_batch_forward_scale_external extract failed ret=%d\n", ret);
        return -1;
    }
    if (out.n != B || out.w != W || out.h != H || out.c != C)
    {
        fprintf(stderr, "test_batch_forward_scale_external shape mismatch\n");
        return -1;
    }

    for (int bi = 0; bi < B; bi++)
    {
        const ncnn::Mat out0 = out.batch(bi);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = out0.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(bi * 100 + q * 10 + i) * (q + 2);
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
                {
                    fprintf(stderr, "test_batch_forward_scale_external mismatch at b=%d q=%d i=%d got %f expect %f\n", bi, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

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
    net.load_param_mem(param_str);

    ncnn::Mat a;
    ncnn::Mat b;
    a.create_batch(4, 3, 2, 3, 4u, 1);
    b.create_batch(4, 3, 2, 2, 4u, 1);

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

    return 0;
}

static int test_batch_forward_split()
{
    const char param_str[] = "7767517\n"
                             "2 3\n"
                             "Input input 0 1 data\n"
                             "Split split 1 2 data out0 out1\n";

    ncnn::Net net;
    net.opt.lightmode = false;
    net.load_param_mem(param_str);

    const int B = 3;
    const int C = 2;
    const int H = 3;
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

    ncnn::Mat out0;
    ncnn::Mat out1;
    int ret0 = ex.extract("out0", out0);
    int ret1 = ex.extract("out1", out1);
    if (ret0 != 0 || ret1 != 0)
    {
        fprintf(stderr, "test_batch_forward_split extract failed ret0=%d ret1=%d\n", ret0, ret1);
        return -1;
    }
    if (out0.n != B || out1.n != B || out0.w != W || out1.w != W || out0.h != H || out1.h != H || out0.c != C || out1.c != C)
    {
        fprintf(stderr, "test_batch_forward_split shape mismatch\n");
        return -1;
    }

    if (CompareMat(input_batch, out0, 1e-5) != 0 || CompareMat(input_batch, out1, 1e-5) != 0)
    {
        fprintf(stderr, "test_batch_forward_split value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_batch_forward_flatten()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Flatten flatten 1 1 data output\n";

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 2;
    const int C = 2;
    const int H = 3;
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
        fprintf(stderr, "test_batch_forward_flatten extract failed ret=%d\n", ret);
        return -1;
    }
    if (output_batch.n != B || output_batch.dims != 1 || output_batch.w != W * H * C)
    {
        fprintf(stderr, "test_batch_forward_flatten shape mismatch\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat sub = output_batch.batch(b);
        const float* ptr = sub;
        for (int i = 0; i < W * H * C; i++)
        {
            float expected = (float)(b * 100 + (i / (W * H)) * 10 + i % (W * H));
            if (!NearlyEqual(ptr[i], expected, 1e-5f))
            {
                fprintf(stderr, "test_batch_forward_flatten mismatch at b=%d i=%d got %f expect %f\n", b, i, ptr[i], expected);
                return -1;
            }
        }
    }

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

    ncnn::Net net;
    net.load_param_mem(param_str);

    const int B = 2;
    const int C = 2;
    const int H = 3;
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
        fprintf(stderr, "test_batch_forward_shape_ops extract failed ret=%d\n", ret);
        return -1;
    }
    if (output_batch.n != B || output_batch.dims != 1 || output_batch.w != W * H * C)
    {
        fprintf(stderr, "test_batch_forward_shape_ops shape mismatch\n");
        return -1;
    }

    for (int b = 0; b < B; b++)
    {
        const ncnn::Mat sub = output_batch.batch(b);
        const float* ptr = sub;
        for (int i = 0; i < W * H * C; i++)
        {
            float expected = (float)(b * 100 + (i / (W * H)) * 10 + i % (W * H));
            if (!NearlyEqual(ptr[i], expected, 1e-5f))
            {
                fprintf(stderr, "test_batch_forward_shape_ops mismatch at b=%d i=%d got %f expect %f\n", b, i, ptr[i], expected);
                return -1;
            }
        }
    }

    return 0;
}

static int test_batch_forward_relu()
{
    // Build a minimal Input -> ReLU network
    // ReLU with slope=0.1 (leaky relu)
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input input 0 1 data\n"
                             "ReLU  relu  1 1 data output 0=1.000000e-01\n";

    ncnn::Net net;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
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
                if (!NearlyEqual(ptr[i], expected, 1e-5f))
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
    const char param_str[] = "7767517\n"
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
            if (!NearlyEqual(ptr[i], expected[i], 1e-5f))
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
            if (!NearlyEqual(ptr[i], expected[i], 1e-5f))
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

static int test_vkmat_create_reset()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();

    ncnn::VkMat m;
    m.create_batch(4, 3, 2, 3, 4u, 1, blob_allocator);
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
    cpu_batch.create_batch(4, 3, 2, 3, 4u, 1);

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
    gpu_batch.create_batch(4, 3, 2, 3, 4u, 1, blob_allocator);
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
                             "Reshape reshape 1 1 data output 0=5 1=3 11=2 2=-1 12=1\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 5;

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
                             "Reshape reshape 1 1 data output 0=5 1=3 2=2 12=2\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
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

static int test_vkmat_batch_forward_reshape_dim_to_batch_axis1()
{
    const char param_str[] = "7767517\n"
                             "2 2\n"
                             "Input   input   0 1 data\n"
                             "Reshape reshape 1 1 data output 0=7 1=5 2=3 12=2 13=1\n";

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
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=3 12=1 13=1\n";

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
    input_batch.create_batch(W, H, C, B, 4u, 1);
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
                             "Reshape reshape 1 1 data output 0=-1 12=1\n";

    const int B = 2;
    const int C = 3;
    const int H = 5;
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
        if (output.elempack != 4)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_batch_to_dim_pack1to4 output should be pack4 fp16s=%d\n", use_fp16_storage);
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
                             "Reshape reshape 1 1 data output 0=7 1=5 11=2 2=4 12=1 13=1\n";

    const int B = 2;
    const int C = 4;
    const int H = 5;
    const int W = 7;

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
        if (output.elempack != 4)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_packed_batch_to_dim_axis1 output should be pack4 fp16s=%d\n", use_fp16_storage);
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
                             "Reshape reshape 1 1 data output 0=5 1=3 2=3 12=2\n";

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
        if (output.elempack != 1)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_dim_to_batch_pack4to1 output should be pack1 fp16s=%d\n", use_fp16_storage);
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
                             "Reshape reshape 1 1 data output 0=7 1=5 2=4 12=2 13=1\n";

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
        if (output.elempack != 4)
        {
            fprintf(stderr, "test_vkmat_batch_forward_reshape_packed_dim_to_batch_axis1 output should be pack4 fp16s=%d\n", use_fp16_storage);
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

static int test_vkmat_batch_forward_relu()
{
    const char param_str[] = "7767517\n"
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
                if (!NearlyEqual(ptr[i], expected, 1e-4f))
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
    const char param_str[] = "7767517\n"
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
            if (!NearlyEqual(ptr[i], expected[i], 1e-4f))
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
            if (!NearlyEqual(ptr[i], expected[i], 1e-4f))
            {
                fprintf(stderr, "test_vkmat_batch_forward_pooling b1 mismatch at i=%d: got %f expect %f\n",
                        i, ptr[i], expected[i]);
                return -1;
            }
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_split()
{
    const char param_str[] = "7767517\n"
                             "2 3\n"
                             "Input input 0 1 data\n"
                             "Split split 1 2 data out0 out1\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.lightmode = false;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    const int B = 3;
    const int C = 2;
    const int H = 3;
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

    ncnn::Mat out0;
    ncnn::Mat out1;
    int ret0 = ex.extract("out0", out0);
    int ret1 = ex.extract("out1", out1);
    if (ret0 != 0 || ret1 != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_split extract failed ret0=%d ret1=%d\n", ret0, ret1);
        return -1;
    }
    if (out0.n != B || out1.n != B || out0.w != W || out1.w != W || out0.h != H || out1.h != H || out0.c != C || out1.c != C)
    {
        fprintf(stderr, "test_vkmat_batch_forward_split shape mismatch\n");
        return -1;
    }

    if (CompareMat(input_batch, out0, 1e-5) != 0 || CompareMat(input_batch, out1, 1e-5) != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_split value mismatch\n");
        return -1;
    }

    return 0;
}

static int test_vkmat_batch_forward_binaryop_same_batch()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input    input0 0 1 a\n"
                             "Input    input1 0 1 b\n"
                             "BinaryOp add    2 1 a b out 0=0\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat a;
    ncnn::Mat b;
    a.create_batch(W, H, C, B, 4u, 1);
    b.create_batch(W, H, C, B, 4u, 1);

    for (int bi = 0; bi < B; bi++)
    {
        ncnn::Mat a0 = a.batch(bi);
        ncnn::Mat b0 = b.batch(bi);
        for (int q = 0; q < C; q++)
        {
            float* pa = a0.channel(q);
            float* pb = b0.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                pa[i] = (float)(bi * 100 + q * 10 + i);
                pb[i] = (float)(bi * 7 + q * 3 + i);
            }
        }
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("a", a);
    ex.input("b", b);

    ncnn::Mat out;
    int ret = ex.extract("out", out);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_binaryop_same_batch extract failed ret=%d\n", ret);
        return -1;
    }
    if (out.n != B || out.w != W || out.h != H || out.c != C)
    {
        fprintf(stderr, "test_vkmat_batch_forward_binaryop_same_batch shape mismatch\n");
        return -1;
    }

    for (int bi = 0; bi < B; bi++)
    {
        const ncnn::Mat out0 = out.batch(bi);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = out0.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(bi * 100 + q * 10 + i + bi * 7 + q * 3 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-4f))
                {
                    fprintf(stderr, "test_vkmat_batch_forward_binaryop_same_batch mismatch at b=%d q=%d i=%d got %f expect %f\n", bi, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_binaryop_broadcast()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input    input0 0 1 a\n"
                             "Input    input1 0 1 b\n"
                             "BinaryOp add    2 1 a b out 0=0\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat a;
    ncnn::Mat b(W, H, C);
    a.create_batch(W, H, C, B, 4u, 1);

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

    ncnn::Extractor ex = net.create_extractor();
    ex.input("a", a);
    ex.input("b", b);

    ncnn::Mat out;
    int ret = ex.extract("out", out);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_binaryop_broadcast extract failed ret=%d\n", ret);
        return -1;
    }
    if (out.n != B || out.w != W || out.h != H || out.c != C)
    {
        fprintf(stderr, "test_vkmat_batch_forward_binaryop_broadcast shape mismatch\n");
        return -1;
    }

    for (int bi = 0; bi < B; bi++)
    {
        const ncnn::Mat out0 = out.batch(bi);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = out0.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(bi * 100 + q * 10 + i + q * 3 + i);
                if (!NearlyEqual(ptr[i], expected, 1e-4f))
                {
                    fprintf(stderr, "test_vkmat_batch_forward_binaryop_broadcast mismatch at b=%d q=%d i=%d got %f expect %f\n", bi, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_scale_external()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input input0 0 1 data\n"
                             "Input input1 0 1 scale\n"
                             "Scale scale0 2 1 data scale out 0=-233\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    const int B = 3;
    const int C = 2;
    const int H = 3;
    const int W = 4;

    ncnn::Mat data;
    data.create_batch(W, H, C, B, 4u, 1);
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

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", data);
    ex.input("scale", scale);

    ncnn::Mat out;
    int ret = ex.extract("out", out);
    if (ret != 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_scale_external extract failed ret=%d\n", ret);
        return -1;
    }
    if (out.n != B || out.w != W || out.h != H || out.c != C)
    {
        fprintf(stderr, "test_vkmat_batch_forward_scale_external shape mismatch\n");
        return -1;
    }

    for (int bi = 0; bi < B; bi++)
    {
        const ncnn::Mat out0 = out.batch(bi);
        for (int q = 0; q < C; q++)
        {
            const float* ptr = out0.channel(q);
            for (int i = 0; i < W * H; i++)
            {
                float expected = (float)(bi * 100 + q * 10 + i) * (q + 2);
                if (!NearlyEqual(ptr[i], expected, 1e-4f))
                {
                    fprintf(stderr, "test_vkmat_batch_forward_scale_external mismatch at b=%d q=%d i=%d got %f expect %f\n", bi, q, i, ptr[i], expected);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_vkmat_batch_forward_binaryop_mismatch()
{
    const char param_str[] = "7767517\n"
                             "3 3\n"
                             "Input    input0 0 1 a\n"
                             "Input    input1 0 1 b\n"
                             "BinaryOp add    2 1 a b out 0=0\n";

    ncnn::Net net;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    net.opt = opt;
    net.load_param_mem(param_str);
    net.load_model((const unsigned char*)"");

    ncnn::Mat a;
    ncnn::Mat b;
    a.create_batch(4, 3, 2, 3, 4u, 1);
    b.create_batch(4, 3, 2, 2, 4u, 1);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("a", a);
    ex.input("b", b);

    ncnn::Mat out;
    int ret = ex.extract("out", out);
    if (ret == 0)
    {
        fprintf(stderr, "test_vkmat_batch_forward_binaryop_mismatch should fail\n");
        return -1;
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
    ret |= test_batch_create_reset();
    ret |= test_batch_reshape();
    ret |= test_batch_reshape_zero_copy();
    ret |= test_batch_reshape_batch_to_dim_flatten();
    ret |= test_batch_reshape_batch_to_dim_4d();
    ret |= test_batch_reshape_dim_to_batch();
    ret |= test_batch_reshape_dim_to_batch_axis1();
    ret |= test_batch_reshape_batch_to_dim_axis1();
    ret |= test_batch_reshape_packed_batch_to_dim_axis0();
    ret |= test_batch_reshape_packed_batch_to_dim_axis0_2d();
    ret |= test_batch_reshape_batch_to_dim_axis0_2d_pack1topacked();
    ret |= test_batch_reshape_packed_batch_to_dim_axis1();
    ret |= test_batch_reshape_packed_batch_to_dim_axis1_2d();
    ret |= test_batch_reshape_batch_to_dim_axis1_pack1topacked();
    ret |= test_batch_reshape_packed_dim_to_batch_axis0();
    ret |= test_batch_reshape_packed_dim_to_batch_axis0_2d();
    ret |= test_batch_reshape_dim_to_batch_axis0_2d_pack1topacked();
    ret |= test_batch_reshape_packed_dim_to_batch_axis1();
    ret |= test_batch_reshape_packed_dim_to_batch_axis1_2d();
    ret |= test_batch_reshape_packed_dim_to_batch_axis1_4d();
    ret |= test_batch_reshape_dim_to_batch_axis1_pack1topacked();
    ret |= test_batch_reshape_batch_to_dim_pack1topacked();
    ret |= test_batch_reshape_batch_to_dim_pack1tohighpack();
    ret |= test_batch_reshape_batch_to_dim_pack4topack1();
    ret |= test_batch_reshape_dim_to_batch_pack1topacked();
    ret |= test_batch_reshape_dim_to_batch_pack1tohighpack();
    ret |= test_batch_reshape_dim_to_batch_pack4topack1();
#if NCNN_BF16
    ret |= test_batch_reshape_bf16_storage_packed();
    ret |= test_batch_reshape_bf16_storage_dim_to_batch_packed();
    ret |= test_batch_reshape_bf16_storage_axis1_packed();
#endif // NCNN_BF16
    ret |= test_batch_reshape_dim_to_batch_no_infer();
    ret |= test_batch_reshape_roundtrip_axis1();
    ret |= test_batch_reshape_roundtrip_axis2();
    ret |= test_batch_reshape_roundtrip();
    ret |= test_batch_reshape_permute_fold();
    ret |= test_batch_reshape_permute_extract();
    ret |= test_backward_compatibility();
    ret |= test_create_batch_single();
    ret |= test_create_batch_1d();
    ret |= test_create_batch_2d();
    ret |= test_batch_forward_binaryop_same_batch();
    ret |= test_batch_forward_binaryop_broadcast();
    ret |= test_batch_forward_scale_external();
    ret |= test_batch_forward_binaryop_mismatch();
    ret |= test_batch_forward_split();
    ret |= test_batch_forward_flatten();
    ret |= test_batch_forward_shape_ops();
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
        ret |= test_vkmat_create_reset();
        ret |= test_vkimage_batch_not_supported();
        ret |= test_vkmat_batch_upload_download();
        ret |= test_vkmat_batch_upload_download_whole();
        ret |= test_vktransfer_batch_upload();
        ret |= test_vkmat_batch_forward_reshape_batch_to_dim();
        ret |= test_vkmat_batch_forward_reshape_dim_to_batch();
        ret |= test_vkmat_batch_forward_reshape_dim_to_batch_axis1();
        ret |= test_vkmat_batch_forward_reshape_batch_to_dim_axis1();
        ret |= test_vkmat_batch_forward_reshape_batch_to_dim_pack1to4();
        ret |= test_vkmat_batch_forward_reshape_packed_batch_to_dim_axis1();
        ret |= test_vkmat_batch_forward_reshape_dim_to_batch_pack4to1();
        ret |= test_vkmat_batch_forward_reshape_packed_dim_to_batch_axis1();
        ret |= test_vkmat_batch_forward_relu();
        ret |= test_vkmat_batch_forward_pooling();
        ret |= test_vkmat_batch_forward_split();
        ret |= test_vkmat_batch_forward_binaryop_same_batch();
        ret |= test_vkmat_batch_forward_binaryop_broadcast();
        ret |= test_vkmat_batch_forward_scale_external();
        ret |= test_vkmat_batch_forward_binaryop_mismatch();
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
