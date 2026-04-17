// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

// Run the Gather layer and return the output blob.
static int run_gather(const ncnn::Mat& data, const ncnn::Mat& indices, int axis, ncnn::Mat& out,
                      int num_threads = 1)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    ncnn::Option opt;
    opt.num_threads = num_threads;
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

// Read index at flat element offset, supporting int32 and int64.
static int read_flat_idx(const ncnn::Mat& m, int flat)
{
    if (m.elemsize == 8)
        return (int)((const int64_t*)(const void*)m)[flat];
    return ((const int*)(const void*)m)[flat];
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
        if (positive_axis == 0)
        {
            for (int y = 0; y < indices.h; y++)
                for (int x = 0; x < idxw; x++)
                {
                    int gi = read_flat_idx(indices, y * idxw + x);
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
                    int gi = read_flat_idx(indices, y * idxw + x);
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
                    int gi = read_flat_idx(indices, (int)(z * i_cstep) + y * idxw + x);
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

static int test_gather(const ncnn::Mat& data, const ncnn::Mat& indices, int axis, const char* name)
{
    ncnn::Mat expected = ref_gather(data, indices, axis);
    ncnn::Mat got;
    if (run_gather(data, indices, axis, got) != 0)
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
    // 1D: out-of-range indices must clamp, not crash.
    ncnn::Mat data = RandomMat(6);
    ncnn::Mat idx;
    idx.create(4, (size_t)4u);
    int* p = (int*)(void*)idx;
    p[0] = -10; // clamps to 0
    p[1] = 0;
    p[2] = 5;
    p[3] = 100; // clamps to 5

    if (test_gather(data, idx, 0, "gather_clamp_1d") != 0) return -1;

    // 2D axis=0: out-of-range row indices
    {
        ncnn::Mat data2d = RandomMat(5, 4); // h=4, w=5
        ncnn::Mat idx2d;
        idx2d.create(5, 3, (size_t)4u); // index shape [3, 5]
        int* q = (int*)(void*)idx2d;
        for (int i = 0; i < 15; i++) q[i] = (i % 3) - 1; // values: -1, 0, 1
        if (test_gather(data2d, idx2d, 0, "gather_clamp_2d_axis0") != 0) return -1;
    }

    // 2D axis=1: out-of-range column indices
    {
        ncnn::Mat data2d = RandomMat(5, 4);
        ncnn::Mat idx2d;
        idx2d.create(3, 4, (size_t)4u);
        int* q = (int*)(void*)idx2d;
        for (int i = 0; i < 12; i++) q[i] = (i % 7) - 1; // includes -1 and 5+
        if (test_gather(data2d, idx2d, 1, "gather_clamp_2d_axis1") != 0) return -1;
    }

    // 3D axis=2: out-of-range indices in the innermost dim
    {
        ncnn::Mat data3d = RandomMat(6, 4, 3);
        ncnn::Mat idx3d;
        idx3d.create(4, 4, 3, (size_t)4u);
        int* q = (int*)(void*)idx3d;
        for (int i = 0; i < (int)idx3d.total(); i++) q[i] = (i % 9) - 2; // includes negatives and overflow
        if (test_gather(data3d, idx3d, 2, "gather_clamp_3d_axis2") != 0) return -1;
    }

    return 0;
}

// Multi-threaded: result must match single-threaded (catches OMP data races).
static int test_gather_multithread()
{
    ncnn::Mat data = RandomMat(16, 12, 8);
    ncnn::Mat idx = make_indices(12, 8, 8, 12); // axis=1 (h=12)

    ncnn::Mat out_single, out_multi;
    if (run_gather(data, idx, 1, out_single, 1) != 0
        || run_gather(data, idx, 1, out_multi, 4) != 0)
    {
        fprintf(stderr, "gather_multithread: forward failed\n");
        return -1;
    }
    return check_equal(out_single, out_multi, "gather_multithread");
}

static int test_gather_int64_indices()
{
    // Verify the int64 index path (elemsize==8) works identically to int32.
    ncnn::Mat data = RandomMat(8, 5); // w=8 h=5

    // 2D axis=0 with int64 indices
    ncnn::Mat idx0_i64 = make_indices_i64(8, 3, 1, 5);
    if (test_gather(data, idx0_i64, 0, "gather_i64_2d_axis0") != 0) return -1;

    // 2D axis=1 with int64 indices
    ncnn::Mat idx1_i64 = make_indices_i64(4, 5, 1, 8);
    if (test_gather(data, idx1_i64, 1, "gather_i64_2d_axis1") != 0) return -1;

    // 3D axis=1 with int64 indices
    ncnn::Mat data3d = RandomMat(8, 6, 4);
    ncnn::Mat idx3d_i64 = make_indices_i64(8, 3, 4, 6);
    if (test_gather(data3d, idx3d_i64, 1, "gather_i64_3d_axis1") != 0) return -1;

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_gather_1d()
           || test_gather_2d()
           || test_gather_3d()
           || test_gather_negative_axis()
           || test_gather_clamp()
           || test_gather_int64_indices()
           || test_gather_multithread();
}
