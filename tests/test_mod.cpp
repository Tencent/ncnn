// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include <math.h>

static int run_mod(const ncnn::Mat& a, const ncnn::Mat& b, int fmode, ncnn::Mat& out)
{
    ncnn::ParamDict pd;
    pd.set(0, fmode);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Mod");
    if (!op)
        return -1;

    op->load_param(pd);

    std::vector<ncnn::Mat> weights(0);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);
    op->create_pipeline(opt);

    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = a;
    bottom_blobs[1] = b;

    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);

    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0)
        return ret;

    out = top_blobs[0];
    return 0;
}

static int test_mod(int w, int h, int c, int fmode, const char* name)
{
    ncnn::Mat a = RandomMat(w, h, c);
    ncnn::Mat b = RandomMat(w, h, c);

    // Ensure b is non-zero (use explicit loops to avoid cstep padding)
    for (int z = 0; z < c; z++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float* bp = (float*)b + z * (int)b.cstep + y * w + x;
                if (*bp == 0.0f) *bp = 1.0f;
            }

    ncnn::Mat out;
    int ret = run_mod(a, b, fmode, out);
    if (ret != 0)
    {
        fprintf(stderr, "%s: forward failed\n", name);
        return -1;
    }

    if (out.w != w || out.h != h || out.c != c)
    {
        fprintf(stderr, "%s: shape mismatch\n", name);
        return -1;
    }

    for (int z = 0; z < c; z++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float val_a = ((const float*)a)[z * (int)a.cstep + y * w + x];
                float val_b = ((const float*)b)[z * (int)b.cstep + y * w + x];
                float val_out = ((const float*)out)[z * (int)out.cstep + y * w + x];

                float expected;
                if (fmode == 0)
                {
                    expected = fmodf(val_a, val_b);
                    if (expected != 0.0f && (val_b < 0.0f) != (expected < 0.0f))
                        expected += val_b;
                }
                else
                {
                    expected = fmodf(val_a, val_b);
                }

                if (fabsf(val_out - expected) > 0.001f)
                {
                    fprintf(stderr, "%s: value mismatch at z=%d y=%d x=%d: got %f expected %f\n",
                            name, z, y, x, val_out, expected);
                    return -1;
                }
            }
    return 0;
}

static int test_mod_negative_values()
{
    // Explicit test with known values: Python-style mod with negative inputs
    ncnn::Mat a(6, (size_t)4u);
    ncnn::Mat b(6, (size_t)4u);
    float avals[6] = {-10, -8, -6, -4, -2, 0};
    float bvals[6] = {3, 3, 3, 3, 3, 3};
    float* ap = a;
    float* bp = b;
    for (int i = 0; i < 6; i++)
    {
        ap[i] = avals[i];
        bp[i] = bvals[i];
    }

    ncnn::Mat out;
    if (run_mod(a, b, 0, out) != 0)
    {
        fprintf(stderr, "test_mod_negative_values: forward failed\n");
        return -1;
    }
    // Python mod: -10%3=2, -8%3=1, -6%3=0, -4%3=2, -2%3=1, 0%3=0
    float expected[6] = {2, 1, 0, 2, 1, 0};
    const float* op_ptr = out;
    for (int i = 0; i < 6; i++)
    {
        if (fabsf(op_ptr[i] - expected[i]) > 0.001f)
        {
            fprintf(stderr, "test_mod_negative_values: mismatch at %d: got %f expected %f\n",
                    i, op_ptr[i], expected[i]);
            return -1;
        }
    }
    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_mod(10, 1, 1, 0, "mod_1d_python")
           || test_mod(10, 1, 1, 1, "mod_1d_c")
           || test_mod(8, 6, 1, 0, "mod_2d")
           || test_mod(4, 6, 8, 0, "mod_3d")
           || test_mod_negative_values();
}
