// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include <limits.h>

static ncnn::Mat int_array(const std::vector<int>& values)
{
    ncnn::Mat m((int)values.size());
    int* ptr = m;
    for (size_t i = 0; i < values.size(); i++)
        ptr[i] = values[i];
    return m;
}

static int test_stridedslice(const ncnn::Mat& a, int start, int end, int step, int mask, bool multi)
{
    const int sizes[4] = {a.w, a.h, a.d, a.c};
    std::vector<int> starts;
    std::vector<int> ends;
    std::vector<int> axes;
    std::vector<int> steps;
    std::vector<int> indices[4];
    int begin_mask = 0;
    int end_mask = 0;
    bool empty = false;
    for (int p = 0; p < 4; p++)
    {
        const bool sliced = p == 0 || (multi && sizes[p] > 1);
        if (sliced)
        {
            int axis = a.dims - 1 - p;
            if (a.dims == 3 && p == 3)
                axis = 0;
            if (mask & 1) begin_mask |= 1 << axes.size();
            if (mask & 2) end_mask |= 1 << axes.size();
            axes.push_back(axis - a.dims);
            starts.push_back(start);
            ends.push_back(end);
            steps.push_back(step);
        }
        int first = start;
        int last = end;
        if (!sliced || (mask & 1))
            first = 0;
        else if (first < 0)
            first += sizes[p];
        if (!sliced || (mask & 2))
            last = sizes[p];
        else if (last < 0)
            last += sizes[p];
        first = std::max(0, first);
        last = std::min(sizes[p], last);
        for (int i = first; i < last; i += sliced ? step : 1)
            indices[p].push_back(i);
        empty |= indices[p].empty();
    }

    ncnn::ParamDict pd;
    pd.set(0, begin_mask);
    pd.set(1, end_mask);
    pd.set(9, int_array(starts));
    pd.set(10, int_array(ends));
    pd.set(11, int_array(axes));
    pd.set(12, int_array(steps));

    if (empty)
    {
        ncnn::Layer* layer = ncnn::create_layer("StridedSlice");
        ncnn::Mat b;
        ncnn::Option opt;
        int ret = layer->load_param(pd);
        if (ret == 0) ret = layer->forward(a, b, opt);
        delete layer;
        return ret != -1;
    }

    ncnn::Mat expected;
    if (a.dims == 1) expected.create((int)indices[0].size());
    if (a.dims == 2) expected.create((int)indices[0].size(), (int)indices[1].size());
    if (a.dims == 3) expected.create((int)indices[0].size(), (int)indices[1].size(), (int)indices[3].size());
    if (a.dims == 4) expected.create((int)indices[0].size(), (int)indices[1].size(), (int)indices[2].size(), (int)indices[3].size());
    for (size_t q = 0; q < indices[3].size(); q++)
    {
        float* outptr = (float*)expected.data + q * expected.cstep;
        for (size_t z = 0; z < indices[2].size(); z++)
            for (size_t y = 0; y < indices[1].size(); y++)
                for (size_t x = 0; x < indices[0].size(); x++)
                    *outptr++ = ((const float*)a.data)[indices[3][q] * a.cstep + (indices[2][z] * a.h + indices[1][y]) * a.w + indices[0][x]];
    }

    ncnn::Layer* layer = ncnn::create_layer("StridedSlice");
    ncnn::Option opt;
    ncnn::Mat b;
    int ret = layer->load_param(pd);
    if (ret == 0) ret = layer->forward(a, b, opt);
    delete layer;
    if (ret == 0) ret = CompareMat(expected, b, 0.f);
    if (ret == 0) ret = test_layer("StridedSlice", pd, std::vector<ncnn::Mat>(), a);
    if (ret != 0)
        fprintf(stderr, "test_stridedslice failed dims=%d shape=(%d %d %d %d) start=%d end=%d step=%d mask=%d multi=%d\n", a.dims, a.w, a.h, a.d, a.c, start, end, step, mask, multi);
    return ret;
}

static int test_stridedslice_invalid()
{
    ncnn::ParamDict pd;
    pd.set(9, int_array(std::vector<int>(1, 0)));
    pd.set(10, int_array(std::vector<int>(1, 10)));
    pd.set(11, int_array(std::vector<int>(1, 0)));
    pd.set(12, int_array(std::vector<int>(1, 0)));
    ncnn::Layer* layer = ncnn::create_layer("StridedSlice");
    int ret = layer->load_param(pd) == 0;
    pd.set(12, int_array(std::vector<int>(1, -1)));
    ret |= layer->load_param(pd) == 0;
    pd.set(12, int_array(std::vector<int>(1, 2)));
    pd.set(10, int_array(std::vector<int>(1, 0)));
    ret |= layer->load_param(pd) != 0;
    ncnn::Mat b;
    ncnn::Option opt;
    ret |= layer->forward(RandomMat(7), b, opt) == 0;
    pd.set(11, int_array(std::vector<int>(1, 2)));
    ret |= layer->load_param(pd) != 0;
    ret |= layer->forward(RandomMat(7), b, opt) == 0;
    pd.set(11, int_array(std::vector<int>(2, 0)));
    ret |= layer->load_param(pd) == 0;
    pd.set(9, int_array(std::vector<int>(2, 0)));
    pd.set(10, int_array(std::vector<int>(2, 5)));
    pd.set(12, int_array(std::vector<int>(2, 2)));
    ret |= layer->load_param(pd) != 0;
    ret |= layer->forward(RandomMat(7, 5), b, opt) == 0;
    pd.set(0, 4);
    ret |= layer->load_param(pd) == 0;
    pd.set(0, 0);
    pd.set(19, "0,0");
    pd.set(20, "1w,1h");
    ret |= layer->load_param(pd) != 0;
    ret |= layer->forward(RandomMat(7, 5), b, opt) == 0;
    delete layer;
    return ret;
}

static int test_stridedslice_expression()
{
    ncnn::ParamDict pd;
    pd.set(11, int_array(std::vector<int>(1, 0)));
    pd.set(12, int_array(std::vector<int>(1, 3)));
    pd.set(19, "2");
    pd.set(20, "-(0w,1)");
    ncnn::Layer* layer = ncnn::create_layer("StridedSlice");
    int ret = layer->load_param(pd);
    const int lengths[] = {8, 12, 11, 7, 13};
    ncnn::Option opt;
    for (int i = 0; i < 5 && ret == 0; i++)
    {
        ncnn::Mat a = RandomMat(lengths[i]);
        ncnn::Mat b;
        ret = layer->forward(a, b, opt);
        const int count = (lengths[i] - 1 - 2 + 2) / 3;
        if (ret != 0 || b.w != count)
        {
            ret = -1;
            break;
        }
        for (int j = 0; j < count; j++)
            if (b[j] != a[2 + j * 3]) ret = -1;
    }
    delete layer;
    return ret;
}

int main()
{
    SRAND(7767517);

    const ncnn::Mat inputs[] = {RandomMat(1), RandomMat(2), RandomMat(3), RandomMat(7), RandomMat(8), RandomMat(301), RandomMat(9, 7), RandomMat(9, 7, 8), RandomMat(9, 7, 5, 8)};
    const int ranges[][4] = {{0, 0, 2, 3}, {1, 0, 2, 2}, {2, 100, 3, 0}, {-5, -1, 2, 0}, {-100, 100, 3, 0}, {0, -233, 2, 0}, {0, INT_MAX, INT_MAX, 0}, {1, 5, 1, 0}};
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++)
        for (size_t j = 0; j < sizeof(ranges) / sizeof(ranges[0]); j++)
            if (test_stridedslice(inputs[i], ranges[j][0], ranges[j][1], ranges[j][2], ranges[j][3], false)
                || test_stridedslice(inputs[i], ranges[j][0], ranges[j][1], ranges[j][2], ranges[j][3], true))
                return -1;

    return test_stridedslice_invalid() || test_stridedslice_expression();
}
