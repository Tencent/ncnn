// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "stridedslice.h"

#include <stdint.h>
#include <string.h>

#include "expression.h"

namespace ncnn {

StridedSlice::StridedSlice()
{
    one_blob_only = true;
    support_inplace = false;
    support_batch = true;
}

int StridedSlice::load_param(const ParamDict& pd)
{
    begin_mask = pd.get(0, 0);
    end_mask = pd.get(1, 0);
    starts = pd.get(9, Mat());
    ends = pd.get(10, Mat());
    axes = pd.get(11, Mat());
    steps = pd.get(12, Mat());
    starts_expr = pd.get(19, "");
    ends_expr = pd.get(20, "");

    const Mat arrays[] = {starts, ends, axes, steps};
    for (int i = 0; i < 4; i++)
    {
        const int type = pd.type(9 + i);
        if (type != 0 && type != 4 && type != 5)
            return -1;
        if (arrays[i].empty())
            continue;
        if (arrays[i].dims != 1 || arrays[i].elempack != 1 || arrays[i].elemsize != 4u)
            return -1;
    }
    const int count = axes.w;
    if (axes.empty() || count < 1 || count > 4 || steps.empty() || steps.w != count)
        return -1;
    if ((starts_expr.empty() && (starts.empty() || starts.w != count)) || (ends_expr.empty() && (ends.empty() || ends.w != count)))
        return -1;
    if (begin_mask < 0 || end_mask < 0 || (begin_mask >> count) || (end_mask >> count))
        return -1;

    const int* axes_ptr = axes;
    const int* steps_ptr = steps;
    for (int i = 0; i < count; i++)
    {
        if (axes_ptr[i] < -4 || axes_ptr[i] > 3 || steps_ptr[i] <= 0)
            return -1;
    }

    const std::string expressions[] = {starts_expr, ends_expr};
    int reference_count = 1;
    for (int i = 0; i < 2; i++)
    {
        if (expressions[i].empty())
            continue;
        int list_size;
        int blob_count;
        bool has_batch;
        if (analyze_list_expression(expressions[i], list_size, blob_count, has_batch) != 0 || list_size != count || has_batch)
            return -1;
        reference_count = std::max(reference_count, blob_count);
    }
    one_blob_only = reference_count == 1;

    return 0;
}

int StridedSlice::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (starts_expr.empty() && ends_expr.empty())
        return forward_slice(bottom_blob, top_blob, starts, ends, opt);

    std::vector<Mat> bottom_blobs(1, bottom_blob);
    std::vector<Mat> top_blobs(1);
    int ret = forward(bottom_blobs, top_blobs, opt);
    if (ret == 0)
        top_blob = top_blobs[0];
    return ret;
}

int StridedSlice::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.empty() || top_blobs.size() != 1)
        return -1;
    if ((size_t)std::max(count_expression_blobs(starts_expr), count_expression_blobs(ends_expr)) > bottom_blobs.size())
        return -1;

    const int* starts_ptr = starts;
    const int* ends_ptr = ends;
    std::vector<int> starts_list;
    std::vector<int> ends_list;
    if (!starts_expr.empty())
    {
        if (eval_list_expression(starts_expr, bottom_blobs, starts_list) != 0 || starts_list.size() != (size_t)axes.w)
            return -1;
        starts_ptr = starts_list.data();
    }
    if (!ends_expr.empty())
    {
        if (eval_list_expression(ends_expr, bottom_blobs, ends_list) != 0 || ends_list.size() != (size_t)axes.w)
            return -1;
        ends_ptr = ends_list.data();
    }

    return forward_slice(bottom_blobs[0], top_blobs[0], starts_ptr, ends_ptr, opt);
}

int StridedSlice::forward_slice(const Mat& bottom_blob, Mat& top_blob, const int* starts_ptr, const int* ends_ptr, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    if (bottom_blob.empty() || dims < 1 || dims > 4 || bottom_blob.elempack != 1)
        return -1;

    // physical order w, h, d, c
    const int sizes[4] = {bottom_blob.w, bottom_blob.h, bottom_blob.d, bottom_blob.c};
    int offsets[4] = {0, 0, 0, 0};
    int strides[4] = {1, 1, 1, 1};
    int outputs[4] = {sizes[0], sizes[1], sizes[2], sizes[3]};
    int used_axes = 0;
    const int* axes_ptr = axes;
    const int* steps_ptr = steps;
    for (int i = 0; i < axes.w; i++)
    {
        int axis = axes_ptr[i];
        if (axis < 0)
            axis += dims;
        if (axis < 0 || axis >= dims || (used_axes & (1 << axis)))
            return -1;
        used_axes |= 1 << axis;
        int physical_axis = dims - 1 - axis;
        if (dims == 3 && axis == 0)
            physical_axis = 3;

        const int64_t length = sizes[physical_axis];
        int64_t start = (begin_mask & (1 << i)) ? 0 : starts_ptr[i];
        int64_t end = (end_mask & (1 << i)) ? length : ends_ptr[i];
        if (start < 0)
            start += length;
        if (end < 0)
            end += length;
        start = std::max((int64_t)0, std::min(start, length));
        end = std::max((int64_t)0, std::min(end, length));
        if (end <= start)
        {
            NCNN_LOGE("StridedSlice does not support empty output");
            return -1;
        }
        offsets[physical_axis] = (int)start;
        strides[physical_axis] = steps_ptr[i];
        outputs[physical_axis] = (int)(1 + (end - start - 1) / steps_ptr[i]);
    }

    // Reference inputs supply shapes, not the output batch size.
    int batch = 1;
#if NCNN_BATCH
    batch = bottom_blob.n;
#endif // NCNN_BATCH
    const size_t elemsize = bottom_blob.elemsize;
    size_t total = elemsize;
    if (batch < 1 || total > (size_t)-1 / batch)
        return -1;
    total *= batch;
    for (int i = 0; i < 4; i++)
    {
        if (outputs[i] <= 0 || total > (size_t)-1 / outputs[i])
            return -1;
        total *= outputs[i];
    }

    if (dims == 1)
        top_blob.create(outputs[0], elemsize, 1, batch, opt.blob_allocator);
    if (dims == 2)
        top_blob.create(outputs[0], outputs[1], elemsize, 1, batch, opt.blob_allocator);
    if (dims == 3)
        top_blob.create(outputs[0], outputs[1], outputs[3], elemsize, 1, batch, opt.blob_allocator);
    if (dims == 4)
        top_blob.create(outputs[0], outputs[1], outputs[2], outputs[3], elemsize, 1, batch, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    for (int b = 0; b < batch; b++)
    {
        const Mat input = bottom_blob.batch(b);
        Mat output = top_blob.batch(b);
#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < outputs[3]; q++)
        {
            const size_t channel_offset = ((size_t)offsets[3] + (size_t)q * strides[3]) * input.cstep;
            unsigned char* outptr = (unsigned char*)output.data + (size_t)q * output.cstep * elemsize;
            for (int z = 0; z < outputs[2]; z++)
            {
                const size_t depth_offset = ((size_t)offsets[2] + (size_t)z * strides[2]) * bottom_blob.h;
                for (int y = 0; y < outputs[1]; y++)
                {
                    const size_t row_offset = (depth_offset + offsets[1] + (size_t)y * strides[1]) * bottom_blob.w;
                    const unsigned char* ptr = (const unsigned char*)input.data + (channel_offset + row_offset + offsets[0]) * elemsize;
                    if (strides[0] == 1)
                    {
                        memcpy(outptr, ptr, (size_t)outputs[0] * elemsize);
                        outptr += (size_t)outputs[0] * elemsize;
                    }
                    else
                    {
                        for (int x = 0; x < outputs[0]; x++)
                        {
                            memcpy(outptr, ptr + (size_t)x * strides[0] * elemsize, elemsize);
                            outptr += elemsize;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
