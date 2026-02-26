// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "topk.h"

#include <algorithm>
#include <vector>

namespace ncnn {

TopK::TopK()
{
    one_blob_only = false;
    support_inplace = false;

    axis = -1;
    largest = 1;
    sorted = 1;
    k = 1;
}

int TopK::load_param(const ParamDict& pd)
{
    axis = pd.get(0, -1);
    largest = pd.get(1, 1);
    sorted = pd.get(2, 1);
    k = pd.get(3, 1);

    return 0;
}

int TopK::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.empty())
        return -1;

    const Mat& bottom_blob = bottom_blobs[0];

    int _k = k;
    if (bottom_blobs.size() >= 2)
    {
        const Mat& k_blob = bottom_blobs[1];
        if (k_blob.total() < 1)
            return -1;

        _k = (int)((const float*)k_blob)[0];
    }

    if (bottom_blob.dims < 1 || bottom_blob.dims > 4)
        return -100;

    int dims = bottom_blob.dims;

    int axis_p = axis < 0 ? axis + dims : axis;
    if (axis_p < 0 || axis_p >= dims)
        return -1;

    int shape[4] = {1, 1, 1, 1};
    shape[0] = bottom_blob.w;
    if (dims >= 2) shape[1] = bottom_blob.h;
    if (dims >= 3) shape[2] = bottom_blob.dims == 3 ? bottom_blob.c : bottom_blob.d;
    if (dims >= 4) shape[3] = bottom_blob.c;

    int axis_size = shape[axis_p];
    if (axis_size <= 0)
        return -1;

    if (_k < 0)
        return -1;
    if (_k > axis_size)
        _k = axis_size;

    int out_shape[4] = {shape[0], shape[1], shape[2], shape[3]};
    out_shape[axis_p] = _k;

    Mat values;
    if (dims == 1) values.create(out_shape[0], 4u, opt.blob_allocator);
    if (dims == 2) values.create(out_shape[0], out_shape[1], 4u, opt.blob_allocator);
    if (dims == 3) values.create(out_shape[0], out_shape[1], out_shape[2], 4u, opt.blob_allocator);
    if (dims == 4) values.create(out_shape[0], out_shape[1], out_shape[2], out_shape[3], 4u, opt.blob_allocator);
    if (values.empty())
        return -100;

    Mat indices;
    if (top_blobs.size() >= 2)
    {
        if (dims == 1) indices.create(out_shape[0], 4u, opt.blob_allocator);
        if (dims == 2) indices.create(out_shape[0], out_shape[1], 4u, opt.blob_allocator);
        if (dims == 3) indices.create(out_shape[0], out_shape[1], out_shape[2], 4u, opt.blob_allocator);
        if (dims == 4) indices.create(out_shape[0], out_shape[1], out_shape[2], out_shape[3], 4u, opt.blob_allocator);
        if (indices.empty())
            return -100;
    }

    const float* ptr = bottom_blob;
    float* outptr = values;
    float* outidxptr = indices;

    int inner = 1;
    for (int i = 0; i < axis_p; i++)
    {
        inner *= shape[i];
    }

    int outer = 1;
    for (int i = axis_p + 1; i < dims; i++)
    {
        outer *= shape[i];
    }

    const bool largest_p = largest != 0;
    const bool sorted_p = sorted != 0;

    const int total_lines = outer * inner;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int line = 0; line < total_lines; line++)
    {
        int outer_i = line / inner;
        int inner_i = line - outer_i * inner;

        int in_base = outer_i * axis_size * inner + inner_i;
        int out_base = outer_i * _k * inner + inner_i;

        std::vector<std::pair<float, int> > vec;
        vec.resize(axis_size);

        for (int j = 0; j < axis_size; j++)
        {
            vec[j].first = ptr[in_base + j * inner];
            vec[j].second = j;
        }

        if (largest_p)
        {
            auto comp = [](const std::pair<float, int>& a, const std::pair<float, int>& b)
            {
                if (a.first != b.first)
                    return a.first > b.first;
                return a.second < b.second;
            };

            if (_k < axis_size)
            {
                if (sorted_p)
                    std::partial_sort(vec.begin(), vec.begin() + _k, vec.end(), comp);
                else
                    std::nth_element(vec.begin(), vec.begin() + _k, vec.end(), comp);
            }
            else
            {
                if (sorted_p)
                    std::sort(vec.begin(), vec.end(), comp);
            }
        }
        else
        {
            auto comp = [](const std::pair<float, int>& a, const std::pair<float, int>& b)
            {
                if (a.first != b.first)
                    return a.first < b.first;
                return a.second < b.second;
            };

            if (_k < axis_size)
            {
                if (sorted_p)
                    std::partial_sort(vec.begin(), vec.begin() + _k, vec.end(), comp);
                else
                    std::nth_element(vec.begin(), vec.begin() + _k, vec.end(), comp);
            }
            else
            {
                if (sorted_p)
                    std::sort(vec.begin(), vec.end(), comp);
            }
        }

        for (int j = 0; j < _k; j++)
        {
            outptr[out_base + j * inner] = vec[j].first;
            if (outidxptr)
                outidxptr[out_base + j * inner] = (float)vec[j].second;
        }
    }

    top_blobs[0] = values;
    if (top_blobs.size() >= 2)
        top_blobs[1] = indices;

    return 0;
}

} // namespace ncnn
