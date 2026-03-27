// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "topk.h"

#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <vector>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

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

#if __ARM_NEON
            if (!output_indices && inner == 1 && axis_size >= 4)
            {
                const float* lineptr = ptr + in_base;

                float best_value = lineptr[0];
                int j = 1;
                int has_nan = topk_isnan(best_value);

                for (; !has_nan && j + 3 < axis_size; j += 4)
                {
                    float32x4_t v = vld1q_f32(lineptr + j);
                    uint32x4_t nan_mask = vmvnq_u32(vceqq_f32(v, v));
                    uint32_t nan_mask_lanes[4];
                    vst1q_u32(nan_mask_lanes, nan_mask);
                    if (nan_mask_lanes[0] || nan_mask_lanes[1] || nan_mask_lanes[2] || nan_mask_lanes[3])
                    {
                        has_nan = 1;
                        break;
                    }

                    float tmp[4];
                    vst1q_f32(tmp, v);

                    if (largest_flag)
                    {
                        if (tmp[0] > best_value) best_value = tmp[0];
                        if (tmp[1] > best_value) best_value = tmp[1];
                        if (tmp[2] > best_value) best_value = tmp[2];
                        if (tmp[3] > best_value) best_value = tmp[3];
                    }
                    else
                    {
                        if (tmp[0] < best_value) best_value = tmp[0];
                        if (tmp[1] < best_value) best_value = tmp[1];
                        if (tmp[2] < best_value) best_value = tmp[2];
                        if (tmp[3] < best_value) best_value = tmp[3];
                    }
                }

                if (!has_nan)
                {
                    for (; j < axis_size; j++)
                    {
                        const float candidate_value = lineptr[j];
                        if (topk_isnan(candidate_value))
                        {
                            has_nan = 1;
                            break;
                        }

                        if (largest_flag)
                        {
                            if (candidate_value > best_value)
                                best_value = candidate_value;
                        }
                        else
                        {
                            if (candidate_value < best_value)
                                best_value = candidate_value;
                        }
                    }
                }

                if (!has_nan)
                {
                    outptr[out_base] = best_value;
                    continue;
                }
            }
#endif // __ARM_NEON

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

    if (_k == axis_size && !sorted_flag)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int line = 0; line < total_lines; line++)
        {
            int outer_i = line / inner;
            int inner_i = line - outer_i * inner;

            int in_base = outer_i * axis_size * inner + inner_i;
            int out_base = outer_i * _k * inner + inner_i;

            if (output_indices)
            {
                for (int j = 0; j < _k; j++)
                {
                    outptr[out_base + j * inner] = ptr[in_base + j * inner];
                    outidxptr[out_base + j * inner] = (float)j;
                }
            }
            else
            {
                for (int j = 0; j < _k; j++)
                {
                    outptr[out_base + j * inner] = ptr[in_base + j * inner];
                }
            }
        }

        top_blobs[0] = values;
        if (top_blobs.size() >= 2)
            top_blobs[1] = indices;

        return 0;
    }

    if (_k <= 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int line = 0; line < total_lines; line++)
        {
            int outer_i = line / inner;
            int inner_i = line - outer_i * inner;

            int in_base = outer_i * axis_size * inner + inner_i;
            int out_base = outer_i * _k * inner + inner_i;

            float top_values[4];
            int top_indices[4];
            int top_count = 0;

            if (sorted_flag)
            {
                for (int j = 0; j < axis_size; j++)
                {
                    const float candidate_value = ptr[in_base + j * inner];

                    if (top_count < _k)
                    {
                        int insert_pos = top_count;
                        while (insert_pos > 0 && topk_value_index_comp(candidate_value, j, top_values[insert_pos - 1], top_indices[insert_pos - 1], largest_flag))
                        {
                            top_values[insert_pos] = top_values[insert_pos - 1];
                            top_indices[insert_pos] = top_indices[insert_pos - 1];
                            insert_pos--;
                        }

                        top_values[insert_pos] = candidate_value;
                        top_indices[insert_pos] = j;
                        top_count++;
                    }
                    else if (topk_value_index_comp(candidate_value, j, top_values[_k - 1], top_indices[_k - 1], largest_flag))
                    {
                        int insert_pos = _k - 1;
                        while (insert_pos > 0 && topk_value_index_comp(candidate_value, j, top_values[insert_pos - 1], top_indices[insert_pos - 1], largest_flag))
                        {
                            top_values[insert_pos] = top_values[insert_pos - 1];
                            top_indices[insert_pos] = top_indices[insert_pos - 1];
                            insert_pos--;
                        }

                        top_values[insert_pos] = candidate_value;
                        top_indices[insert_pos] = j;
                    }
                }
            }
            else
            {
                for (int j = 0; j < axis_size; j++)
                {
                    const float candidate_value = ptr[in_base + j * inner];

                    if (top_count < _k)
                    {
                        top_values[top_count] = candidate_value;
                        top_indices[top_count] = j;
                        top_count++;
                    }
                    else
                    {
                        int worst_pos = 0;
                        for (int t = 1; t < _k; t++)
                        {
                            if (topk_value_index_comp(top_values[worst_pos], top_indices[worst_pos], top_values[t], top_indices[t], largest_flag))
                                worst_pos = t;
                        }

                        if (topk_value_index_comp(candidate_value, j, top_values[worst_pos], top_indices[worst_pos], largest_flag))
                        {
                            top_values[worst_pos] = candidate_value;
                            top_indices[worst_pos] = j;
                        }
                    }
                }
            }

            if (output_indices)
            {
                for (int j = 0; j < _k; j++)
                {
                    outptr[out_base + j * inner] = top_values[j];
                    outidxptr[out_base + j * inner] = (float)top_indices[j];
                }
            }
            else
            {
                for (int j = 0; j < _k; j++)
                {
                    outptr[out_base + j * inner] = top_values[j];
                }
            }
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
