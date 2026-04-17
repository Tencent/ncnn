// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "topk.h"

#include <stdint.h>
#include <string.h>

#if NCNN_SIMPLESTL
#include "simplestl.h"
#else
#include <algorithm>
#include <vector>
#endif

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

// Fast comparison assuming both values are non-NaN (common case).
static inline bool topk_value_index_comp_nonnan(float a_value, int a_index, float b_value, int b_index, bool largest)
{
    if (a_value != b_value)
        return largest ? (a_value > b_value) : (a_value < b_value);
    return a_index < b_index;
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

        const size_t k_elemsize = k_blob.elemsize / k_blob.elempack;
        if (k_elemsize == 8)
            _k = (int)((const int64_t*)(const void*)k_blob)[0];
        else if (k_elemsize == 4)
            _k = ((const int*)(const void*)k_blob)[0];
        else
            return -1;
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

    if (_k == 0)
    {
        // Return empty (zero-sized) output blobs without allocation
        top_blobs[0] = Mat();
        if (top_blobs.size() >= 2)
            top_blobs[1] = Mat();
        return 0;
    }

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

    const float* ptr = bottom_blob;
    float* outptr = values;
    int* outidxptr = (int*)(void*)(indices.data);
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

    // ncnn 3-/4-D mats have a channel stride (cstep) that may be larger than w*h
    // due to alignment padding.  The flat inner/outer indexing must account for this:
    //   - when axis reduces a non-channel dim, the outer loop spans channels and
    //     the channel offset must use cstep rather than the product of spatial sizes;
    //   - when axis IS the channel dim, the per-element j-stride must be cstep.
    const size_t in_cstep = (dims >= 3) ? (size_t)bottom_blob.cstep : 0;
    const size_t out_cstep = (dims >= 3) ? values.cstep : 0;
    const bool axis_is_channel = (dims >= 3 && positive_axis == dims - 1);
    // spatial-only outer count: channels factored out so cstep can be used separately
    const int c_channels = (!axis_is_channel && dims >= 3) ? shape[dims - 1] : 1;
    const int outer_spatial = (dims >= 3 && !axis_is_channel) ? outer / c_channels : outer;
    // stride when stepping along the axis in memory
    const size_t in_axis_stride = axis_is_channel ? in_cstep : (size_t)inner;
    const size_t out_axis_stride = axis_is_channel ? out_cstep : (size_t)inner;

    if (_k == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int line = 0; line < total_lines; line++)
        {
            int outer_i = line / inner;
            int inner_i = line - outer_i * inner;

            size_t in_base, out_base;
            if (!axis_is_channel && dims >= 3)
            {
                const int ci = outer_i / outer_spatial;
                const int sp_i = outer_i % outer_spatial;
                in_base = (size_t)ci * in_cstep + (size_t)sp_i * axis_size * inner + inner_i;
                out_base = (size_t)ci * out_cstep + (size_t)sp_i * 1 * inner + inner_i;
            }
            else
            {
                in_base = (size_t)outer_i * axis_size * inner + inner_i;
                out_base = (size_t)outer_i * 1 * inner + inner_i;
            }

#if __ARM_NEON
            if (!output_indices && inner == 1 && axis_size >= 4)
            {
                const float* lineptr = ptr + in_base;
                int has_nan = topk_isnan(lineptr[0]);

                // Accumulate best4 across all NEON chunks; reduce to scalar only once.
                float32x4_t best4 = vdupq_n_f32(lineptr[0]);
                int j = 1;

                for (; !has_nan && j + 3 < axis_size; j += 4)
                {
                    float32x4_t v = vld1q_f32(lineptr + j);
                    // NaN check: v != v is true for NaN; OR all lanes via 64-bit view
                    uint32x4_t nan_mask = vmvnq_u32(vceqq_f32(v, v));
                    uint64x2_t nm64 = vreinterpretq_u64_u32(nan_mask);
                    if (vgetq_lane_u64(nm64, 0) | vgetq_lane_u64(nm64, 1))
                    {
                        has_nan = 1;
                        break;
                    }

                    best4 = largest_flag ? vmaxq_f32(best4, v) : vminq_f32(best4, v);
                }

                if (!has_nan)
                {
                    // Reduce best4 to scalar once after the loop
                    float32x2_t m = largest_flag
                        ? vpmax_f32(vget_low_f32(best4), vget_high_f32(best4))
                        : vpmin_f32(vget_low_f32(best4), vget_high_f32(best4));
                    m = largest_flag ? vpmax_f32(m, m) : vpmin_f32(m, m);
                    float best_value = vget_lane_f32(m, 0);

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

                    if (!has_nan)
                    {
                        outptr[out_base] = best_value;
                        continue;
                    }
                }
            }
#endif // __ARM_NEON

            float best_value = ptr[in_base];
            int best_index = 0;

            // Fast path: no NaN check per comparison pair (common case).
            // topk_value_index_comp checks both operands for NaN on every call;
            // here we check only the candidate, and fall back only when NaN is found.
            bool has_nan = topk_isnan(best_value);
            if (!has_nan)
            {
                if (largest_flag)
                {
                    for (int j = 1; j < axis_size; j++)
                    {
                        const float v = ptr[in_base + j * in_axis_stride];
                        if (topk_isnan(v)) { has_nan = true; break; }
                        if (v > best_value) { best_value = v; best_index = j; }
                    }
                }
                else
                {
                    for (int j = 1; j < axis_size; j++)
                    {
                        const float v = ptr[in_base + j * in_axis_stride];
                        if (topk_isnan(v)) { has_nan = true; break; }
                        if (v < best_value) { best_value = v; best_index = j; }
                    }
                }
            }
            if (has_nan)
            {
                // NaN-aware fallback: NaN sorts last, ties broken by index.
                best_value = ptr[in_base];
                best_index = 0;
                for (int j = 1; j < axis_size; j++)
                {
                    const float v = ptr[in_base + j * in_axis_stride];
                    if (topk_value_index_comp(v, j, best_value, best_index, largest_flag))
                    {
                        best_value = v;
                        best_index = j;
                    }
                }
            }

            outptr[out_base] = best_value;
            if (output_indices)
                outidxptr[out_base] = best_index;
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

            size_t in_base, out_base;
            if (!axis_is_channel && dims >= 3)
            {
                const int ci = outer_i / outer_spatial;
                const int sp_i = outer_i % outer_spatial;
                in_base = (size_t)ci * in_cstep + (size_t)sp_i * axis_size * inner + inner_i;
                out_base = (size_t)ci * out_cstep + (size_t)sp_i * _k * inner + inner_i;
            }
            else
            {
                in_base = (size_t)outer_i * axis_size * inner + inner_i;
                out_base = (size_t)outer_i * _k * inner + inner_i;
            }

            if (output_indices)
            {
                for (int j = 0; j < _k; j++)
                {
                    outptr[out_base + j * out_axis_stride] = ptr[in_base + j * in_axis_stride];
                    outidxptr[out_base + j * out_axis_stride] = j;
                }
            }
            else
            {
                for (int j = 0; j < _k; j++)
                {
                    outptr[out_base + j * out_axis_stride] = ptr[in_base + j * in_axis_stride];
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

            size_t in_base, out_base;
            if (!axis_is_channel && dims >= 3)
            {
                const int ci = outer_i / outer_spatial;
                const int sp_i = outer_i % outer_spatial;
                in_base = (size_t)ci * in_cstep + (size_t)sp_i * axis_size * inner + inner_i;
                out_base = (size_t)ci * out_cstep + (size_t)sp_i * _k * inner + inner_i;
            }
            else
            {
                in_base = (size_t)outer_i * axis_size * inner + inner_i;
                out_base = (size_t)outer_i * _k * inner + inner_i;
            }

            float top_values[4];
            int top_indices[4];
            int top_count = 0;

            // has_nan_in_top: tracks whether the current top-k buffer contains any NaN.
            // When false, use the cheaper non-NaN comparator in the insertion sort.
            bool has_nan_in_top = false;

            if (sorted_flag)
            {
                for (int j = 0; j < axis_size; j++)
                {
                    const float candidate_value = ptr[in_base + j * in_axis_stride];
                    const bool cand_nan = topk_isnan(candidate_value);

                    // Select comparator: skip NaN handling when neither side has NaN.
                    #define COMP_K4(a_v, a_i, b_v, b_i) \
                        ((!cand_nan && !has_nan_in_top) \
                            ? topk_value_index_comp_nonnan(a_v, a_i, b_v, b_i, largest_flag) \
                            : topk_value_index_comp(a_v, a_i, b_v, b_i, largest_flag))

                    if (top_count < _k)
                    {
                        int insert_pos = top_count;
                        while (insert_pos > 0 && COMP_K4(candidate_value, j, top_values[insert_pos - 1], top_indices[insert_pos - 1]))
                        {
                            top_values[insert_pos] = top_values[insert_pos - 1];
                            top_indices[insert_pos] = top_indices[insert_pos - 1];
                            insert_pos--;
                        }

                        top_values[insert_pos] = candidate_value;
                        top_indices[insert_pos] = j;
                        top_count++;
                        if (cand_nan) has_nan_in_top = true;
                    }
                    else if (COMP_K4(candidate_value, j, top_values[_k - 1], top_indices[_k - 1]))
                    {
                        if (!cand_nan && has_nan_in_top)
                        {
                            // Evicting a NaN: recheck whether any NaN remains in top buffer.
                            has_nan_in_top = false;
                            for (int t = 0; t < _k - 1; t++)
                                if (topk_isnan(top_values[t])) { has_nan_in_top = true; break; }
                        }

                        int insert_pos = _k - 1;
                        while (insert_pos > 0 && COMP_K4(candidate_value, j, top_values[insert_pos - 1], top_indices[insert_pos - 1]))
                        {
                            top_values[insert_pos] = top_values[insert_pos - 1];
                            top_indices[insert_pos] = top_indices[insert_pos - 1];
                            insert_pos--;
                        }

                        top_values[insert_pos] = candidate_value;
                        top_indices[insert_pos] = j;
                        if (cand_nan) has_nan_in_top = true;
                    }

                    #undef COMP_K4
                }
            }
            else
            {
                for (int j = 0; j < axis_size; j++)
                {
                    const float candidate_value = ptr[in_base + j * in_axis_stride];
                    const bool cand_nan = topk_isnan(candidate_value);

                    if (top_count < _k)
                    {
                        top_values[top_count] = candidate_value;
                        top_indices[top_count] = j;
                        top_count++;
                        if (cand_nan) has_nan_in_top = true;
                    }
                    else
                    {
                        const bool use_fast = (!cand_nan && !has_nan_in_top);
                        int worst_pos = 0;
                        for (int t = 1; t < _k; t++)
                        {
                            bool is_worse = use_fast
                                ? topk_value_index_comp_nonnan(top_values[worst_pos], top_indices[worst_pos], top_values[t], top_indices[t], largest_flag)
                                : topk_value_index_comp(top_values[worst_pos], top_indices[worst_pos], top_values[t], top_indices[t], largest_flag);
                            if (is_worse) worst_pos = t;
                        }

                        bool replace = use_fast
                            ? topk_value_index_comp_nonnan(candidate_value, j, top_values[worst_pos], top_indices[worst_pos], largest_flag)
                            : topk_value_index_comp(candidate_value, j, top_values[worst_pos], top_indices[worst_pos], largest_flag);

                        if (replace)
                        {
                            if (!cand_nan && has_nan_in_top)
                            {
                                has_nan_in_top = false;
                                for (int t = 0; t < _k; t++)
                                    if (t != worst_pos && topk_isnan(top_values[t])) { has_nan_in_top = true; break; }
                            }
                            top_values[worst_pos] = candidate_value;
                            top_indices[worst_pos] = j;
                            if (cand_nan) has_nan_in_top = true;
                        }
                    }
                }
            }

            if (output_indices)
            {
                for (int j = 0; j < _k; j++)
                {
                    outptr[out_base + j * out_axis_stride] = top_values[j];
                    outidxptr[out_base + j * out_axis_stride] = top_indices[j];
                }
            }
            else
            {
                for (int j = 0; j < _k; j++)
                {
                    outptr[out_base + j * out_axis_stride] = top_values[j];
                }
            }
        }

        top_blobs[0] = values;
        if (top_blobs.size() >= 2)
            top_blobs[1] = indices;

        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int line = 0; line < total_lines; line++)
    {
        // Reuse thread-local scratch to avoid one malloc/free per line.
#if !NCNN_SIMPLESTL
        static thread_local std::vector<std::pair<float, int> > tl_vec;
        tl_vec.resize(axis_size);
        std::vector<std::pair<float, int> >& vec = tl_vec;
#else
        std::vector<std::pair<float, int> > vec(axis_size);
#endif

        topk_pair_comparator comp(largest_flag);

        int outer_i = line / inner;
        int inner_i = line - outer_i * inner;

        size_t in_base, out_base;
        if (!axis_is_channel && dims >= 3)
        {
            const int ci = outer_i / outer_spatial;
            const int sp_i = outer_i % outer_spatial;
            in_base = (size_t)ci * in_cstep + (size_t)sp_i * axis_size * inner + inner_i;
            out_base = (size_t)ci * out_cstep + (size_t)sp_i * _k * inner + inner_i;
        }
        else
        {
            in_base = (size_t)outer_i * axis_size * inner + inner_i;
            out_base = (size_t)outer_i * _k * inner + inner_i;
        }

        for (int j = 0; j < axis_size; j++)
        {
            vec[j].first = ptr[in_base + j * in_axis_stride];
            vec[j].second = j;
        }

        if (_k < axis_size)
        {
#if NCNN_SIMPLESTL
            std::partial_sort(vec.begin(), vec.begin() + _k, vec.end(), comp);
#else
            if (sorted_flag)
            {
                std::nth_element(vec.begin(), vec.begin() + _k, vec.end(), comp);
                std::sort(vec.begin(), vec.begin() + _k, comp);
            }
            else
                std::nth_element(vec.begin(), vec.begin() + _k, vec.end(), comp);
#endif
        }
        else
        {
            if (sorted_flag)
#if NCNN_SIMPLESTL
                std::partial_sort(vec.begin(), vec.end(), vec.end(), comp);
#else
                std::sort(vec.begin(), vec.end(), comp);
#endif
        }

        if (output_indices)
        {
            for (int j = 0; j < _k; j++)
            {
                outptr[out_base + j * out_axis_stride] = vec[j].first;
                outidxptr[out_base + j * out_axis_stride] = vec[j].second;
            }
        }
        else
        {
            for (int j = 0; j < _k; j++)
            {
                outptr[out_base + j * out_axis_stride] = vec[j].first;
            }
        }
    }

    top_blobs[0] = values;
    if (top_blobs.size() >= 2)
        top_blobs[1] = indices;

    return 0;
}

} // namespace ncnn
