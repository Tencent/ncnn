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

// Compare layer output against fmodf reference with exact equality.
// The impl uses ::fmodf (float-precision), so results must be bit-identical.
static int test_mod(int w, int h, int c, int fmode, const char* name)
{
    ncnn::Mat a = RandomMat(w, h, c);
    ncnn::Mat b = RandomMat(w, h, c);

    // Ensure b is non-zero
    for (int z = 0; z < c; z++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float* bp = (float*)b + z * (int)b.cstep + y * w + x;
                if (*bp == 0.0f) *bp = 1.0f;
            }

    ncnn::Mat out;
    if (run_mod(a, b, fmode, out) != 0)
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

                if (val_out != expected)
                {
                    fprintf(stderr, "%s: value mismatch at z=%d y=%d x=%d: got %f expected %f\n",
                            name, z, y, x, val_out, expected);
                    return -1;
                }
            }
    return 0;
}

// Zero divisor: b=0 must return 0, not crash.
static int test_mod_zero_divisor()
{
    ncnn::Mat a(5, (size_t)4u);
    ncnn::Mat b(5, (size_t)4u);
    float* ap = a; float* bp = b;
    ap[0] = 7.f; ap[1] = -3.f; ap[2] = 0.f; ap[3] = 100.f; ap[4] = -50.f;
    for (int i = 0; i < 5; i++) bp[i] = 0.0f;

    ncnn::Mat out;
    for (int fmode = 0; fmode <= 1; fmode++)
    {
        if (run_mod(a, b, fmode, out) != 0)
        {
            fprintf(stderr, "test_mod_zero_divisor fmode=%d: forward failed\n", fmode);
            return -1;
        }
        const float* op = out;
        for (int i = 0; i < 5; i++)
        {
            if (op[i] != 0.0f)
            {
                fprintf(stderr, "test_mod_zero_divisor fmode=%d: expected 0 at %d, got %f\n",
                        fmode, i, op[i]);
                return -1;
            }
        }
    }
    return 0;
}

// Python-style mod with known negative inputs/divisors.
static int test_mod_negative_values()
{
    ncnn::Mat a(6, (size_t)4u);
    ncnn::Mat b(6, (size_t)4u);
    float avals[6] = {-10, -8, -6, -4, -2, 0};
    float bvals[6] = {3, 3, 3, 3, 3, 3};
    float* ap = a; float* bp = b;
    for (int i = 0; i < 6; i++) { ap[i] = avals[i]; bp[i] = bvals[i]; }

    ncnn::Mat out;
    if (run_mod(a, b, 0, out) != 0)
    {
        fprintf(stderr, "test_mod_negative_values: forward failed\n");
        return -1;
    }
    // Python mod: -10%3=2, -8%3=1, -6%3=0, -4%3=2, -2%3=1, 0%3=0
    float expected[6] = {2, 1, 0, 2, 1, 0};
    const float* op = out;
    for (int i = 0; i < 6; i++)
    {
        if (op[i] != expected[i])
        {
            fprintf(stderr, "test_mod_negative_values: mismatch at %d: got %f expected %f\n",
                    i, op[i], expected[i]);
            return -1;
        }
    }
    return 0;
}

// C-style fmod with negative b — sign of result follows the dividend, not divisor.
static int test_mod_fmod1_negative_b()
{
    ncnn::Mat a(4, (size_t)4u);
    ncnn::Mat b(4, (size_t)4u);
    float* ap = a; float* bp = b;
    ap[0] = 7.f;  bp[0] = -3.f;  // fmod(7, -3)  = 1  (sign of dividend +7)
    ap[1] = -7.f; bp[1] = 3.f;   // fmod(-7, 3)  = -1 (sign of dividend -7)
    ap[2] = -7.f; bp[2] = -3.f;  // fmod(-7, -3) = -1
    ap[3] = 6.f;  bp[3] = -2.f;  // fmod(6, -2)  = 0

    ncnn::Mat out;
    if (run_mod(a, b, 1, out) != 0)
    {
        fprintf(stderr, "test_mod_fmod1_negative_b: forward failed\n");
        return -1;
    }
    const float* op = out;
    float expected[4] = {
        fmodf(7.f, -3.f),
        fmodf(-7.f, 3.f),
        fmodf(-7.f, -3.f),
        fmodf(6.f, -2.f)
    };
    for (int i = 0; i < 4; i++)
    {
        if (op[i] != expected[i])
        {
            fprintf(stderr, "test_mod_fmod1_negative_b: mismatch at %d: got %f expected %f\n",
                    i, op[i], expected[i]);
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
           || test_mod(8, 6, 1, 0, "mod_2d_python")
           || test_mod(8, 6, 1, 1, "mod_2d_c")
           || test_mod(4, 6, 8, 0, "mod_3d_python")
           || test_mod(4, 6, 8, 1, "mod_3d_c")
           || test_mod_zero_divisor()
           || test_mod_negative_values()
           || test_mod_fmod1_negative_b();
}
