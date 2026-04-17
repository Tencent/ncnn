// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include <string.h>

static int run_expand(const ncnn::Mat& data, const ncnn::Mat& shape, ncnn::Mat& out)
{
    ncnn::ParamDict pd;

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Expand");
    if (!op)
        return -1;

    op->load_param(pd);

    std::vector<ncnn::Mat> weights(0);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);
    op->create_pipeline(opt);

    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = data;
    bottom_blobs[1] = shape;

    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);

    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0)
        return ret;

    out = top_blobs[0];
    return 0;
}

// Build a 1D int32 shape Mat in ncnn ordering (w, h, c).
static ncnn::Mat make_shape_i32(int w, int h, int c)
{
    ncnn::Mat s(3, (size_t)4u);
    int* p = (int*)(void*)s;
    p[0] = w;
    p[1] = h;
    p[2] = c;
    return s;
}

// Build a 1D int64 shape Mat (same values, different elemsize).
static ncnn::Mat make_shape_i64(int w, int h, int c)
{
    ncnn::Mat s(3, (size_t)8u);
    int64_t* p = (int64_t*)(void*)s;
    p[0] = w;
    p[1] = h;
    p[2] = c;
    return s;
}

static int check_equal(const ncnn::Mat& a, const ncnn::Mat& b, const char* name)
{
    if (a.dims != b.dims || a.w != b.w || a.h != b.h || a.c != b.c)
    {
        fprintf(stderr, "%s: shape mismatch got(%d %d %d dims=%d) expected(%d %d %d dims=%d)\n",
                name, a.w, a.h, a.c, a.dims, b.w, b.h, b.c, b.dims);
        return -1;
    }
    const float* ap = a;
    const float* bp = b;
    for (int z = 0; z < a.c; z++)
        for (int y = 0; y < a.h; y++)
            for (int x = 0; x < a.w; x++)
            {
                float got = ap[(int)(z * a.cstep) + y * a.w + x];
                float exp = bp[(int)(z * b.cstep) + y * b.w + x];
                if (got != exp)
                {
                    fprintf(stderr, "%s: value mismatch at [%d,%d,%d]: got %f expected %f\n",
                            name, z, y, x, got, exp);
                    return -1;
                }
            }
    return 0;
}

static ncnn::Mat ref_expand(const ncnn::Mat& src, int out_w, int out_h, int out_c)
{
    ncnn::Mat out;
    out.create(out_w, out_h, out_c, (size_t)4u);

    const float* sp = src;
    float* op = out;

    for (int z = 0; z < out_c; z++)
    {
        int sz = (src.c > 1) ? z : 0;
        const float* sc = sp + sz * (int)src.cstep;
        float* dc = op + z * (int)out.cstep;
        for (int y = 0; y < out_h; y++)
        {
            int sy = (src.h > 1) ? y : 0;
            const float* sr = sc + sy * src.w;
            float* dr = dc + y * out_w;
            for (int x = 0; x < out_w; x++)
            {
                int sx = (src.w > 1) ? x : 0;
                dr[x] = sr[sx];
            }
        }
    }
    return out;
}

static int test_expand(const ncnn::Mat& data, int out_w, int out_h, int out_c, const char* name)
{
    ncnn::Mat shape = make_shape_i32(out_w, out_h, out_c);
    ncnn::Mat expected = ref_expand(data, out_w, out_h, out_c);
    ncnn::Mat got;
    if (run_expand(data, shape, got) != 0)
    {
        fprintf(stderr, "%s: forward failed\n", name);
        return -1;
    }
    return check_equal(got, expected, name);
}

// --- Tests ---

static int test_expand_scalar_to_1d()
{
    ncnn::Mat data = RandomMat(1, 1, 1);
    return test_expand(data, 10, 1, 1, "expand_scalar_to_w10");
}

static int test_expand_broadcast_w()
{
    // in_w=1 → out_w=5: exercises the scalar broadcast fill path (out_w < 16)
    ncnn::Mat data = RandomMat(1, 3, 1);
    return test_expand(data, 5, 3, 1, "expand_broadcast_w");
}

static int test_expand_broadcast_w_neon()
{
    // in_w=1 → out_w=20: out_w >= 16 triggers the NEON 4×-unrolled fill path
    ncnn::Mat data = RandomMat(1, 4, 1);
    return test_expand(data, 20, 4, 1, "expand_broadcast_w_neon");
}

static int test_expand_broadcast_h()
{
    ncnn::Mat data = RandomMat(4, 1, 1);
    return test_expand(data, 4, 6, 1, "expand_broadcast_h");
}

static int test_expand_broadcast_c()
{
    ncnn::Mat data = RandomMat(4, 3, 1);
    return test_expand(data, 4, 3, 8, "expand_broadcast_c");
}

static int test_expand_broadcast_wh()
{
    // Broadcasts both w and h simultaneously
    ncnn::Mat data = RandomMat(1, 1, 3);
    return test_expand(data, 8, 5, 3, "expand_broadcast_wh");
}

static int test_expand_full_broadcast()
{
    ncnn::Mat data = RandomMat(1, 1, 1);
    return test_expand(data, 4, 6, 8, "expand_full_broadcast");
}

static int test_expand_no_broadcast()
{
    ncnn::Mat data = RandomMat(4, 3, 2);
    return test_expand(data, 4, 3, 2, "expand_no_broadcast");
}

static int test_expand_1d_to_3d()
{
    ncnn::Mat data = RandomMat(4);
    return test_expand(data, 4, 6, 8, "expand_1d_to_3d");
}

static int test_expand_2d_to_3d()
{
    ncnn::Mat data = RandomMat(4, 3);
    return test_expand(data, 4, 3, 8, "expand_2d_to_3d");
}

// int64 shape blob — exercises the shape_is_int64 branch in Expand::forward.
static int test_expand_int64_shape()
{
    ncnn::Mat data = RandomMat(1, 2, 1);
    ncnn::Mat shape = make_shape_i64(6, 2, 4);
    ncnn::Mat expected = ref_expand(data, 6, 2, 4);
    ncnn::Mat got;
    if (run_expand(data, shape, got) != 0)
    {
        fprintf(stderr, "expand_int64_shape: forward failed\n");
        return -1;
    }
    return check_equal(got, expected, "expand_int64_shape");
}

// -1 in shape means "keep that dimension" (tgt_dim <= 0 branch).
static int test_expand_negative_one_shape()
{
    ncnn::Mat data = RandomMat(4, 3, 2);
    // shape = (-1, -1, -1) should return data unchanged
    ncnn::Mat shape = make_shape_i32(-1, -1, -1);
    ncnn::Mat got;
    if (run_expand(data, shape, got) != 0)
    {
        fprintf(stderr, "expand_negative_one_shape: forward failed\n");
        return -1;
    }
    return check_equal(got, data, "expand_negative_one_shape");
}

int main()
{
    SRAND(7767517);

    return 0
           || test_expand_scalar_to_1d()
           || test_expand_broadcast_w()
           || test_expand_broadcast_w_neon()
           || test_expand_broadcast_h()
           || test_expand_broadcast_c()
           || test_expand_broadcast_wh()
           || test_expand_full_broadcast()
           || test_expand_no_broadcast()
           || test_expand_1d_to_3d()
           || test_expand_2d_to_3d()
           || test_expand_int64_shape()
           || test_expand_negative_one_shape();
}
