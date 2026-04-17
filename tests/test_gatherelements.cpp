// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

// Run the GatherElements layer and return the output blob.
static int run_gatherelements(const ncnn::Mat& data, const ncnn::Mat& indices, int axis, ncnn::Mat& out)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("GatherElements");
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

// Read index at flat element offset, supporting int32 and int64.
static int read_flat_idx(const ncnn::Mat& m, int flat)
{
    if (m.elemsize == 8)
        return (int)((const int64_t*)(const void*)m)[flat];
    return ((const int*)(const void*)m)[flat];
}

// Reference GatherElements: PyTorch-style axis ordering.
// Index has same rank as data. For each position (z,y,x) in index:
//   axis=0: out[z,y,x] = data[idx[z,y,x], y, x]
//   axis=1: out[z,y,x] = data[z, idx[z,y,x], x]
//   axis=2: out[z,y,x] = data[z, y, idx[z,y,x]]
static ncnn::Mat ref_gatherelements(const ncnn::Mat& data, const ncnn::Mat& indices, int axis)
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
    float* op_ptr = out;

    if (dims == 1)
    {
        for (int x = 0; x < indices.w; x++)
        {
            int gi = read_flat_idx(indices, x);
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
        for (int y = 0; y < indices.h; y++)
            for (int x = 0; x < idxw; x++)
            {
                int gi = read_flat_idx(indices, y * idxw + x);
                if (gi < 0) gi += axis_size;
                if (gi < 0) gi = 0;
                if (gi >= axis_size) gi = axis_size - 1;

                int flat_in = (positive_axis == 0) ? gi * dw + x : y * dw + gi;
                op_ptr[y * out.w + x] = dp[flat_in];
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
                    int gi = read_flat_idx(indices, (int)(z * i_cstep) + y * idxw + x);
                    if (gi < 0) gi += axis_size;
                    if (gi < 0) gi = 0;
                    if (gi >= axis_size) gi = axis_size - 1;

                    int flat_in;
                    if (positive_axis == 0)
                        flat_in = (int)(gi * d_cstep) + y * dw + x;
                    else if (positive_axis == 1)
                        flat_in = (int)(z * d_cstep) + gi * dw + x;
                    else
                        flat_in = (int)(z * d_cstep) + y * dw + gi;

                    op_ptr[(int)(z * o_cstep) + y * out.w + x] = dp[flat_in];
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

// Build an int64 index Mat with the same pattern.
static ncnn::Mat make_indices_i64(int w, int h, int c, int axis_size)
{
    ncnn::Mat m;
    if (c > 1)
        m.create(w, h, c, (size_t)8u);
    else if (h > 1)
        m.create(w, h, (size_t)8u);
    else
        m.create(w, (size_t)8u);

    int64_t* p = (int64_t*)(void*)m;
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
    // Use explicit loops to avoid comparing uninitialized cstep padding bytes
    const float* ad = (const float*)a.data;
    const float* bd = (const float*)b.data;
    for (int z = 0; z < a.c; z++)
        for (int y = 0; y < a.h; y++)
            for (int x = 0; x < a.w; x++)
            {
                float av = ad[z * a.cstep + y * a.w + x];
                float bv = bd[z * b.cstep + y * b.w + x];
                if (av != bv)
                {
                    fprintf(stderr, "%s: value mismatch at z=%d y=%d x=%d: got %f expected %f\n",
                            name, z, y, x, av, bv);
                    return -1;
                }
            }
    return 0;
}

static int test_gatherelements(const ncnn::Mat& data, const ncnn::Mat& indices, int axis, const char* name)
{
    ncnn::Mat expected = ref_gatherelements(data, indices, axis);
    ncnn::Mat got;
    int ret = run_gatherelements(data, indices, axis, got);
    if (ret != 0)
    {
        fprintf(stderr, "%s: forward failed\n", name);
        return -1;
    }
    return check_equal(got, expected, name);
}

static int test_gatherelements_1d()
{
    ncnn::Mat data = RandomMat(10);
    ncnn::Mat idx = make_indices(5, 1, 1, 10);
    return test_gatherelements(data, idx, 0, "gatherelements_1d_axis0");
}

static int test_gatherelements_2d()
{
    ncnn::Mat data = RandomMat(8, 5); // w=8 h=5

    // axis=0 (h, size=5), index shape [3,8]
    ncnn::Mat idx0 = make_indices(8, 3, 1, 5);
    if (test_gatherelements(data, idx0, 0, "gatherelements_2d_axis0") != 0) return -1;

    // axis=1 (w, size=8), index shape [5,4]
    ncnn::Mat idx1 = make_indices(4, 5, 1, 8);
    if (test_gatherelements(data, idx1, 1, "gatherelements_2d_axis1") != 0) return -1;

    return 0;
}

static int test_gatherelements_3d()
{
    ncnn::Mat data = RandomMat(8, 6, 4); // w=8 h=6 c=4

    // axis=0 (c, size=4), index shape [2,6,8]
    ncnn::Mat idx0 = make_indices(8, 6, 2, 4);
    if (test_gatherelements(data, idx0, 0, "gatherelements_3d_axis0") != 0) return -1;

    // axis=1 (h, size=6), index shape [4,3,8]
    ncnn::Mat idx1 = make_indices(8, 3, 4, 6);
    if (test_gatherelements(data, idx1, 1, "gatherelements_3d_axis1") != 0) return -1;

    // axis=2 (w, size=8), index shape [4,6,5]
    ncnn::Mat idx2 = make_indices(5, 6, 4, 8);
    if (test_gatherelements(data, idx2, 2, "gatherelements_3d_axis2") != 0) return -1;

    return 0;
}

static int test_gatherelements_negative_axis()
{
    ncnn::Mat data = RandomMat(8, 6, 4); // w=8 h=6 c=4

    // axis=-1 == axis=2 (w, size=8)
    ncnn::Mat idx = make_indices(5, 6, 4, 8);
    if (test_gatherelements(data, idx, -1, "gatherelements_3d_axis-1") != 0) return -1;

    // axis=-3 == axis=0 (c, size=4)
    ncnn::Mat idx0 = make_indices(8, 6, 2, 4);
    if (test_gatherelements(data, idx0, -3, "gatherelements_3d_axis-3") != 0) return -1;

    return 0;
}

static int test_gatherelements_clamp()
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

    return test_gatherelements(data, idx, 0, "gatherelements_clamp");
}

static int test_gatherelements_int64_indices()
{
    // Verify the int64 index path (elemsize==8) works identically to int32.
    ncnn::Mat data = RandomMat(8, 5); // w=8 h=5

    ncnn::Mat idx0_i64 = make_indices_i64(8, 3, 1, 5);
    if (test_gatherelements(data, idx0_i64, 0, "gatherelements_i64_2d_axis0") != 0) return -1;

    ncnn::Mat idx1_i64 = make_indices_i64(4, 5, 1, 8);
    if (test_gatherelements(data, idx1_i64, 1, "gatherelements_i64_2d_axis1") != 0) return -1;

    ncnn::Mat data3d = RandomMat(8, 6, 4);
    ncnn::Mat idx3d_i64 = make_indices_i64(8, 3, 4, 6);
    if (test_gatherelements(data3d, idx3d_i64, 1, "gatherelements_i64_3d_axis1") != 0) return -1;

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_gatherelements_1d()
           || test_gatherelements_2d()
           || test_gatherelements_3d()
           || test_gatherelements_negative_axis()
           || test_gatherelements_clamp()
           || test_gatherelements_int64_indices();
}
