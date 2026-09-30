// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

static int test_concat(const std::vector<ncnn::Mat>& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); //axis

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Concat", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_concat failed a[0].dims=%d a[0]=(%d %d %d %d) axis=%d\n", a[0].dims, a[0].w, a[0].h, a[0].d, a[0].c, axis);
    }

    return ret;
}

static int test_concat_0()
{
    ncnn::Mat a[] = {
        RandomMat(15, 5, 6, 13),
        RandomMat(15, 5, 6, 20),
        RandomMat(15, 5, 6, 24),
        RandomMat(15, 5, 6, 48)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                std::vector<ncnn::Mat> as(4);
                as[0] = a[i];
                as[1] = a[j];
                as[2] = a[k];
                as[3] = a[k];

                int ret = test_concat(as, 0) || test_concat(as, -4);
                if (ret != 0)
                    return ret;
            }
        }
    }

    return 0;
}

static int test_concat_1()
{
    ncnn::Mat a[] = {
        RandomMat(15, 3, 15, 13),
        RandomMat(15, 3, 16, 20),
        RandomMat(15, 3, 17, 24),
        RandomMat(15, 3, 18, 48)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        std::vector<ncnn::Mat> as(3);
        as[0] = a[i];
        as[1] = a[i];
        as[2] = a[i];

        int ret = test_concat(as, 1) || test_concat(as, -3);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_concat_2()
{
    ncnn::Mat a[] = {
        RandomMat(15, 15, 6, 13),
        RandomMat(15, 16, 6, 20),
        RandomMat(15, 17, 6, 24),
        RandomMat(15, 18, 6, 48)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        std::vector<ncnn::Mat> as(3);
        as[0] = a[i];
        as[1] = a[i];
        as[2] = a[i];

        int ret = test_concat(as, 2) || test_concat(as, -2);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_concat_3()
{
    ncnn::Mat a[] = {
        RandomMat(15, 5, 7, 13),
        RandomMat(16, 5, 7, 20),
        RandomMat(17, 5, 7, 24),
        RandomMat(18, 5, 7, 48)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        std::vector<ncnn::Mat> as(3);
        as[0] = a[i];
        as[1] = a[i];
        as[2] = a[i];

        int ret = test_concat(as, 3) || test_concat(as, -1);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_concat_4()
{
    ncnn::Mat a[] = {
        RandomMat(15, 13, 13),
        RandomMat(15, 13, 20),
        RandomMat(15, 13, 24),
        RandomMat(15, 13, 48)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                std::vector<ncnn::Mat> as(4);
                as[0] = a[i];
                as[1] = a[j];
                as[2] = a[k];
                as[3] = a[k];

                int ret = test_concat(as, 0) || test_concat(as, -3);
                if (ret != 0)
                    return ret;
            }
        }
    }

    return 0;
}

static int test_concat_5()
{
    ncnn::Mat a[] = {
        RandomMat(15, 15, 13),
        RandomMat(15, 16, 20),
        RandomMat(15, 17, 24),
        RandomMat(15, 18, 48)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        std::vector<ncnn::Mat> as(3);
        as[0] = a[i];
        as[1] = a[i];
        as[2] = a[i];

        int ret = test_concat(as, 1) || test_concat(as, -2);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_concat_6()
{
    ncnn::Mat a[] = {
        RandomMat(15, 13, 13),
        RandomMat(16, 13, 20),
        RandomMat(17, 13, 24),
        RandomMat(18, 13, 48)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        std::vector<ncnn::Mat> as(3);
        as[0] = a[i];
        as[1] = a[i];
        as[2] = a[i];

        int ret = test_concat(as, 2) || test_concat(as, -1);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_concat_7()
{
    ncnn::Mat a[] = {
        RandomMat(19, 29),
        RandomMat(19, 44),
        RandomMat(19, 56),
        RandomMat(19, 80)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                std::vector<ncnn::Mat> as(4);
                as[0] = a[i];
                as[1] = a[j];
                as[2] = a[k];
                as[3] = a[k];

                int ret = test_concat(as, 0) || test_concat(as, -2);
                if (ret != 0)
                    return ret;
            }
        }
    }

    return 0;
}

static int test_concat_8()
{
    ncnn::Mat a[] = {
        RandomMat(19, 29),
        RandomMat(16, 44),
        RandomMat(17, 56),
        RandomMat(18, 80)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        std::vector<ncnn::Mat> as(3);
        as[0] = a[i];
        as[1] = a[i];
        as[2] = a[i];

        int ret = test_concat(as, 1) || test_concat(as, -1);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_concat_9()
{
    ncnn::Mat a[] = {
        RandomMat(29),
        RandomMat(44),
        RandomMat(56),
        RandomMat(80)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                std::vector<ncnn::Mat> as(4);
                as[0] = a[i];
                as[1] = a[j];
                as[2] = a[k];
                as[3] = a[k];

                int ret = test_concat(as, 0) || test_concat(as, -1);
                if (ret != 0)
                    return ret;
            }
        }
    }

    return 0;
}

// #7025: mismatched non-concat dims must fail cleanly instead of heap-overflow.
static int test_concat_shape_mismatch_case(const ncnn::Mat& a, const ncnn::Mat& b, int axis)
{
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = a;
    bottom_blobs[1] = b;
    std::vector<ncnn::Mat> top_blobs(1);

    ncnn::ParamDict pd;
    pd.set(0, axis);

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* op = ncnn::create_layer("Concat");
    if (!op)
    {
        fprintf(stderr, "test_concat_shape_mismatch create_layer failed\n");
        return -1;
    }

    op->load_param(pd);
    op->create_pipeline(opt);

    const int ret = op->forward(bottom_blobs, top_blobs, opt);

    op->destroy_pipeline(opt);
    delete op;

    if (ret != -1 || !top_blobs[0].empty())
    {
        fprintf(stderr, "test_concat_shape_mismatch expected failure, got success\n");
        return -1;
    }

    return 0;
}

static int test_concat_shape_mismatch()
{
    const ncnn::Mat a[] = {
        RandomMat(12, 7, 128),
        RandomMat(5, 3), RandomMat(5, 3),
        RandomMat(5, 3, 4), RandomMat(5, 3, 4),
        RandomMat(5, 3, 2, 4), RandomMat(5, 3, 2, 4),
        RandomMat(5, 3, 2, 4), RandomMat(5, 3, 2, 4)
    };
    const ncnn::Mat b[] = {
        RandomMat(7, 12, 128),
        RandomMat(6, 3), RandomMat(5, 4),
        RandomMat(6, 3, 4), RandomMat(5, 4, 4),
        RandomMat(5, 3, 3, 4), RandomMat(6, 3, 2, 4),
        RandomMat(5, 3, 2, 5), RandomMat(5, 4, 2, 4)
    };
    const int axes[] = {0, 0, 1, 1, 2, 0, 1, 2, 3};
    for (size_t i = 0; i < sizeof(axes) / sizeof(axes[0]); i++)
    {
        if (test_concat_shape_mismatch_case(a[i], b[i], axes[i])
            || test_concat_shape_mismatch_case(a[i], b[i], axes[i] - a[i].dims))
            return -1;
    }
    // Different ranks and out-of-range axes must also fail before allocation.
    return test_concat_shape_mismatch_case(RandomMat(5), RandomMat(5, 3), 0)
           || test_concat_shape_mismatch_case(a[0], a[0], 3)
           || test_concat_shape_mismatch_case(a[0], a[0], -4);
}

// No malformed dispatch is submitted to the device.
#if NCNN_VULKAN
static int test_concat_vulkan_shape_mismatch()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::Layer* op = ncnn::create_layer_vulkan("Concat");
    if (!op)
        return -1;

    op->vkdev = vkdev;
    ncnn::ParamDict pd;
    pd.set(0, 0);
    op->load_param(pd);

    ncnn::VkBlobAllocator allocator(vkdev);
    ncnn::Option opt;
    opt.blob_vkallocator = &allocator;
    opt.workspace_vkallocator = &allocator;
    opt.use_fp16_storage = false;
    opt.use_fp16_packed = false;
    if (op->create_pipeline(opt) != 0)
    {
        op->destroy_pipeline(opt);
        delete op;
        return -1;
    }

    int ret;
    bool output_empty;
    {
        std::vector<ncnn::VkMat> bottom_blobs(2);
        bottom_blobs[0].create(12, 7, 128, 4u, 1, &allocator);
        bottom_blobs[1].create(7, 12, 128, 4u, 1, &allocator);
        if (bottom_blobs[0].empty() || bottom_blobs[1].empty())
        {
            op->destroy_pipeline(opt);
            delete op;
            return -1;
        }
        std::vector<ncnn::VkMat> top_blobs(1);
        ncnn::VkCompute cmd(vkdev);
        ret = op->forward(bottom_blobs, top_blobs, cmd, opt);
        output_empty = top_blobs[0].empty();
    }
    op->destroy_pipeline(opt);
    delete op;

    if (ret != -1 || !output_empty)
    {
        fprintf(stderr, "test_concat_vulkan_shape_mismatch expected rejection before output allocation\n");
        return -1;
    }
    return 0;
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    return 0
#if NCNN_VULKAN
           || test_concat_vulkan_shape_mismatch()
#endif // NCNN_VULKAN
           || test_concat_0()
           || test_concat_1()
           || test_concat_2()
           || test_concat_3()
           || test_concat_4()
           || test_concat_5()
           || test_concat_6()
           || test_concat_7()
           || test_concat_8()
           || test_concat_9()
           || test_concat_shape_mismatch();
}
