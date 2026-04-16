// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

// Run the Gather layer and return the output blob.
static int run_gather(const ncnn::Mat& data, const ncnn::Mat& indices, int axis, ncnn::Mat& out)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Gather");
    if (!op)
        return -1;

    op->load_param(pd);

    std::vector<ncnn::Mat> weights(0);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);
    op->create_pipeline(opt);

    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = data;
    bottom_blobs[1] = indices;

    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);

    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0)
        return ret;

    out = top_blobs[0];
    return 0;
}

// Reference gather: PyTorch-style axis ordering (axis=0 = outermost).
// 1D axis=0:  out[x]     = data[idx[x]]
// 2D axis=0:  out[y,x]   = data[idx[y,x], x]
// 2D axis=1:  out[y,x]   = data[y, idx[y,x]]
// 3D axis=0:  out[z,y,x] = data[idx[z,y,x], y, x]
// 3D axis=1:  out[z,y,x] = data[z, idx[z,y,x], x]
// 3D axis=2:  out[z,y,x] = data[z, y, idx[z,y,x]]
static ncnn::Mat ref_gather(const ncnn::Mat& data, const ncnn::Mat& indices, int axis)
{
    const int dims = data.dims;
    int positive_axis = axis < 0 ? axis + dims : axis;

    int shape[3] = {1, 1, 1};
    if (dims == 1)
        shape[0] = data.w;
    else if (dims == 2)
    {
        shape[0] = data.h;
        shape[1] = data.w;
    }
    else
    {
        shape[0] = data.c;
        shape[1] = data.h;
        shape[2] = data.w;
    }
    const int axis_size = shape[positive_axis];

    ncnn::Mat out;
    if (indices.dims == 1)
        out.create(indices.w, (size_t)4u);
    else if (indices.dims == 2)
        out.create(indices.w, indices.h, (size_t)4u);
    else
        out.create(indices.w, indices.h, indices.c, (size_t)4u);

    const float* dp = data;
    const int* ip = (const int*)(const void*)indices;
    float* op_ptr = out;

    if (dims == 1)
    {
        for (int x = 0; x < indices.w; x++)
        {
            int gi = ip[x];
            if (gi < 0) gi += axis_size;
            if (gi < 0) gi = 0;
            if (gi >= axis_size) gi = axis_size - 1;
            op_ptr[x] = dp[gi];
        }
    }
    else if (dims == 2)
    {
        const int dw = data.w;
        const int idxw = indices.w;
        if (positive_axis == 0)
        {
            for (int y = 0; y < indices.h; y++)
                for (int x = 0; x < idxw; x++)
                {
                    int gi = ip[y * idxw + x];
                    if (gi < 0) gi += axis_size;
                    if (gi < 0) gi = 0;
                    if (gi >= axis_size) gi = axis_size - 1;
                    op_ptr[y * out.w + x] = dp[gi * dw + x];
                }
        }
        else
        {
            for (int y = 0; y < indices.h; y++)
                for (int x = 0; x < idxw; x++)
                {
                    int gi = ip[y * idxw + x];
                    if (gi < 0) gi += axis_size;
                    if (gi < 0) gi = 0;
                    if (gi >= axis_size) gi = axis_size - 1;
                    op_ptr[y * out.w + x] = dp[y * dw + gi];
                }
        }
    }
    else // dims == 3
    {
        const int dw = data.w;
        const size_t d_cstep = data.cstep;
        const size_t i_cstep = indices.cstep;
        const size_t o_cstep = out.cstep;
        const int idxw = indices.w;

        for (int z = 0; z < indices.c; z++)
            for (int y = 0; y < indices.h; y++)
                for (int x = 0; x < idxw; x++)
                {
                    int gi = ip[(int)(z * i_cstep) + y * idxw + x];
                    if (gi < 0) gi += axis_size;
                    if (gi < 0) gi = 0;
                    if (gi >= axis_size) gi = axis_size - 1;

                    float val;
                    if (positive_axis == 0)
                        val = dp[(int)(gi * d_cstep) + y * dw + x];
                    else if (positive_axis == 1)
                        val = dp[(int)(z * d_cstep) + gi * dw + x];
                    else
                        val = dp[(int)(z * d_cstep) + y * dw + gi];

                    op_ptr[(int)(z * o_cstep) + y * out.w + x] = val;
                }
    }

    return out;
}

// Build an int32 index Mat with values in [0, axis_size).
// Uses a deterministic pattern: idx[i] = (i * 3 + 1) % axis_size.
static ncnn::Mat make_indices(int w, int h, int c, int axis_size)
{
    ncnn::Mat m;
    if (c > 1)
        m.create(w, h, c, (size_t)4u);
    else if (h > 1)
        m.create(w, h, (size_t)4u);
    else
        m.create(w, (size_t)4u);

    int* p = (int*)(void*)m;
    int total = (int)m.total();
    for (int i = 0; i < total; i++)
        p[i] = (i * 3 + 1) % axis_size;
    return m;
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
    int total = (int)a.total();
    for (int i = 0; i < total; i++)
    {
        if (ap[i] != bp[i])
        {
            fprintf(stderr, "%s: value mismatch at %d: got %f expected %f\n", name, i, ap[i], bp[i]);
            return -1;
        }
    }
    return 0;
}

static int test_gather(const ncnn::Mat& data, const ncnn::Mat& indices, int axis, const char* name)
{
    ncnn::Mat expected = ref_gather(data, indices, axis);
    ncnn::Mat got;
    int ret = run_gather(data, indices, axis, got);
    if (ret != 0)
    {
        fprintf(stderr, "%s: forward failed\n", name);
        return -1;
    }
    return check_equal(got, expected, name);
}

static int test_gather_1d()
{
    ncnn::Mat data = RandomMat(10);
    ncnn::Mat idx = make_indices(5, 1, 1, 10);
    return test_gather(data, idx, 0, "gather_1d_axis0");
}

static int test_gather_2d()
{
    ncnn::Mat data = RandomMat(8, 5); // w=8 h=5

    // axis=0 (PyTorch outermost = h, size=5), index shape [3,8]
    ncnn::Mat idx0 = make_indices(8, 3, 1, 5);
    if (test_gather(data, idx0, 0, "gather_2d_axis0") != 0) return -1;

    // axis=1 (PyTorch innermost = w, size=8), index shape [5,4]
    ncnn::Mat idx1 = make_indices(4, 5, 1, 8);
    if (test_gather(data, idx1, 1, "gather_2d_axis1") != 0) return -1;

    return 0;
}

static int test_gather_3d()
{
    ncnn::Mat data = RandomMat(8, 6, 4); // w=8 h=6 c=4

    // axis=0 (c, size=4), index shape [2,6,8]
    ncnn::Mat idx0 = make_indices(8, 6, 2, 4);
    if (test_gather(data, idx0, 0, "gather_3d_axis0") != 0) return -1;

    // axis=1 (h, size=6), index shape [4,3,8]
    ncnn::Mat idx1 = make_indices(8, 3, 4, 6);
    if (test_gather(data, idx1, 1, "gather_3d_axis1") != 0) return -1;

    // axis=2 (w, size=8), index shape [4,6,5]
    ncnn::Mat idx2 = make_indices(5, 6, 4, 8);
    if (test_gather(data, idx2, 2, "gather_3d_axis2") != 0) return -1;

    return 0;
}

static int test_gather_negative_axis()
{
    ncnn::Mat data = RandomMat(8, 6, 4); // w=8 h=6 c=4

    // axis=-1 == axis=2 (w, size=8)
    ncnn::Mat idx = make_indices(5, 6, 4, 8);
    if (test_gather(data, idx, -1, "gather_3d_axis-1") != 0) return -1;

    // axis=-3 == axis=0 (c, size=4)
    ncnn::Mat idx0 = make_indices(8, 6, 2, 4);
    if (test_gather(data, idx0, -3, "gather_3d_axis-3") != 0) return -1;

    return 0;
}

static int test_gather_clamp()
{
    // Verify that out-of-range indices are clamped, not crashed.
    ncnn::Mat data = RandomMat(6);
    ncnn::Mat idx;
    idx.create(4, (size_t)4u);
    int* p = (int*)(void*)idx;
    p[0] = -10; // clamps to 0
    p[1] = 0;
    p[2] = 5;
    p[3] = 100; // clamps to 5

    return test_gather(data, idx, 0, "gather_clamp");
}

int main()
{
    SRAND(7767517);

    return 0
           || test_gather_1d()
           || test_gather_2d()
           || test_gather_3d()
           || test_gather_negative_axis()
           || test_gather_clamp();
}
