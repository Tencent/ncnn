// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "topk.h"

#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <vector>

namespace ncnn {

static inline bool topk_isnan(float v)
{
    uint32_t u;
    memcpy(&u, &v, sizeof(uint32_t));
    return (u & 0x7fffffff) > 0x7f800000;
}

static inline bool topk_pair_comp(const std::pair<float, int>& a, const std::pair<float, int>& b, bool largest)
{
    const bool a_nan = topk_isnan(a.first);
    const bool b_nan = topk_isnan(b.first);

    // Keep NaN at the end for both largest/smallest to ensure deterministic ordering.
    if (a_nan || b_nan)
    {
        if (a_nan != b_nan)
            return !a_nan && b_nan;

        return a.second < b.second;
    }

    if (a.first != b.first)
        return largest ? (a.first > b.first) : (a.first < b.first);

    return a.second < b.second;
}

static inline bool topk_value_index_comp(float a_value, int a_index, float b_value, int b_index, bool largest)
{
    const bool a_nan = topk_isnan(a_value);
    const bool b_nan = topk_isnan(b_value);

    if (a_nan || b_nan)
    {
        if (a_nan != b_nan)
            return !a_nan && b_nan;

        return a_index < b_index;
    }

    if (a_value != b_value)
        return largest ? (a_value > b_value) : (a_value < b_value);

    return a_index < b_index;
}

struct topk_pair_comparator
{
    topk_pair_comparator(bool _largest)
        : largest(_largest)
    {
    }

    bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) const
    {
        return topk_pair_comp(a, b, largest);
    }

    bool largest;
};

TopK::TopK()
{
    one_blob_only = false;
    support_inplace = false;
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

    const int dims = bottom_blob.dims;

    const int positive_axis = axis < 0 ? axis + dims : axis;
    if (positive_axis < 0 || positive_axis >= dims)
        return -1;

    int shape[4] = {1, 1, 1, 1};
    shape[0] = bottom_blob.w;
    if (dims >= 2) shape[1] = bottom_blob.h;
    if (dims >= 3) shape[2] = bottom_blob.dims == 3 ? bottom_blob.c : bottom_blob.d;
    if (dims >= 4) shape[3] = bottom_blob.c;

    const int axis_size = shape[positive_axis];
    if (axis_size <= 0)
        return -1;

    if (_k < 0)
        return -1;
    if (_k > axis_size)
        _k = axis_size;

    int out_shape[4] = {shape[0], shape[1], shape[2], shape[3]};
    out_shape[positive_axis] = _k;

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

    if (_k == 0)
    {
        top_blobs[0] = values;
        if (top_blobs.size() >= 2)
            top_blobs[1] = indices;

        return 0;
    }

    const float* ptr = bottom_blob;
    float* outptr = values;
    float* outidxptr = indices;
    const bool output_indices = outidxptr != 0;

    int inner = 1;
    for (int i = 0; i < positive_axis; i++)
    {
        inner *= shape[i];
    }

    int outer = 1;
    for (int i = positive_axis + 1; i < dims; i++)
    {
        outer *= shape[i];
    }

    const bool largest_flag = largest != 0;
    const bool sorted_flag = sorted != 0;

    const int total_lines = outer * inner;

    if (_k == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int line = 0; line < total_lines; line++)
        {
            int outer_i = line / inner;
            int inner_i = line - outer_i * inner;

            int in_base = outer_i * axis_size * inner + inner_i;
            int out_base = outer_i * inner + inner_i;

            float best_value = ptr[in_base];
            int best_index = 0;

            for (int j = 1; j < axis_size; j++)
            {
                const float candidate_value = ptr[in_base + j * inner];
                if (topk_value_index_comp(candidate_value, j, best_value, best_index, largest_flag))
                {
                    best_value = candidate_value;
                    best_index = j;
                }
            }

            outptr[out_base] = best_value;
            if (output_indices)
                outidxptr[out_base] = (float)best_index;
        }

        top_blobs[0] = values;
        if (top_blobs.size() >= 2)
            top_blobs[1] = indices;

        return 0;
    }

    #pragma omp parallel num_threads(opt.num_threads)
    {
        std::vector<std::pair<float, int> > vec;
        vec.resize(axis_size);

        topk_pair_comparator comp(largest_flag);

        #pragma omp for
        for (int line = 0; line < total_lines; line++)
        {
            int outer_i = line / inner;
            int inner_i = line - outer_i * inner;

            int in_base = outer_i * axis_size * inner + inner_i;
            int out_base = outer_i * _k * inner + inner_i;

            for (int j = 0; j < axis_size; j++)
            {
                vec[j].first = ptr[in_base + j * inner];
                vec[j].second = j;
            }

            if (_k < axis_size)
            {
                if (sorted_flag)
                {
                    std::nth_element(vec.begin(), vec.begin() + _k, vec.end(), comp);
                    std::sort(vec.begin(), vec.begin() + _k, comp);
                }
                else
                    std::nth_element(vec.begin(), vec.begin() + _k, vec.end(), comp);
            }
            else
            {
                if (sorted_flag)
                    std::sort(vec.begin(), vec.end(), comp);
            }

            if (output_indices)
            {
                for (int j = 0; j < _k; j++)
                {
                    outptr[out_base + j * inner] = vec[j].first;
                    outidxptr[out_base + j * inner] = (float)vec[j].second;
                }
            }
            else
            {
                for (int j = 0; j < _k; j++)
                {
                    outptr[out_base + j * inner] = vec[j].first;
                }
            }
        }
    }

    top_blobs[0] = values;
    if (top_blobs.size() >= 2)
        top_blobs[1] = indices;

    return 0;
}

} // namespace ncnn
