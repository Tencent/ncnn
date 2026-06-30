// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape.h"

#include "expression.h"

#include <string.h>

namespace ncnn {

Reshape::Reshape()
{
    one_blob_only = true;
    support_inplace = false;
#if NCNN_BATCH
    input_batch_axis = 233;
    output_batch_axis = 233;
#endif
}

int Reshape::load_param(const ParamDict& pd)
{
    w = pd.get(0, -233);
    h = pd.get(1, -233);
    d = pd.get(11, -233);
    c = pd.get(2, -233);

    ndim = 4;
    if (d == -233)
        ndim = 3;
    if (c == -233)
        ndim = 2;
    if (h == -233)
        ndim = 1;
    if (w == -233)
        ndim = 0;

#if NCNN_BATCH
    input_batch_axis = pd.get(12, 233);
    output_batch_axis = pd.get(13, 233);
    if (input_batch_axis != 233 || output_batch_axis != 233)
    {
        support_batch = true;
    }
#else
    if (pd.get(12, 233) != 233 || pd.get(13, 233) != 233)
    {
        NCNN_LOGE("please build ncnn with NCNN_BATCH enabled for batch inference");
        return -1;
    }
#endif

    shape_expr = pd.get(6, "");

    // count reference blobs
    if (!shape_expr.empty())
    {
        const int blob_count = count_expression_blobs(shape_expr);
        if (blob_count > 1)
            one_blob_only = false;

        // resolve ndim from expression
        std::vector<Mat> blobs(blob_count);
        std::vector<int> outshape;
        int er = eval_list_expression(shape_expr, blobs, outshape);
        if (er != 0)
            return -1;

        ndim = (int)outshape.size();
    }

    return 0;
}

int Reshape::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_blobs(1);
    bottom_blobs[0] = bottom_blob;
    std::vector<Mat> top_blobs(1);
    int ret = forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];
    return ret;
}

#if NCNN_BATCH
static size_t get_batch_reshape_offset(const Mat& m, const int* shape, int dims, int batch_axis, size_t i)
{
    int coord[5] = {0, 0, 0, 0, 0};
    for (int j = dims - 1; j >= 0; j--)
    {
        coord[j] = (int)(i % shape[j]);
        i /= shape[j];
    }

    int b = 0;
    int p[4] = {0, 0, 0, 0};
    int pdims = 0;
    for (int j = 0; j < dims; j++)
    {
        if (j == batch_axis)
        {
            b = coord[j];
            continue;
        }

        p[pdims++] = coord[j];
    }

    size_t offset = (size_t)b * m.nstep;
    if (pdims == 1)
        offset += p[0];
    if (pdims == 2)
        offset += (size_t)p[0] * m.w + p[1];
    if (pdims == 3)
        offset += (size_t)p[0] * m.cstep + (size_t)p[1] * m.w + p[2];
    if (pdims == 4)
        offset += (size_t)p[0] * m.cstep + (size_t)p[1] * m.w * m.h + (size_t)p[2] * m.w + p[3];

    return offset;
}
#endif // NCNN_BATCH

int Reshape::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    // resolve out shape
    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

#if NCNN_BATCH
    if (!shape_expr.empty() && input_batch_axis == 233 && output_batch_axis == 233)
#else
    if (!shape_expr.empty())
#endif
    {
        eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
    }

#if NCNN_BATCH
    if (input_batch_axis != 233 || output_batch_axis != 233)
    {
        if (bottom_blob.elempack != 1)
            return -1;

        int physical_input_shape[4] = {0, 0, 0, 0};
        if (bottom_blob.dims == 1)
            physical_input_shape[0] = bottom_blob.w;
        if (bottom_blob.dims == 2)
        {
            physical_input_shape[0] = bottom_blob.h;
            physical_input_shape[1] = bottom_blob.w;
        }
        if (bottom_blob.dims == 3)
        {
            physical_input_shape[0] = bottom_blob.c;
            physical_input_shape[1] = bottom_blob.h;
            physical_input_shape[2] = bottom_blob.w;
        }
        if (bottom_blob.dims == 4)
        {
            physical_input_shape[0] = bottom_blob.c;
            physical_input_shape[1] = bottom_blob.d;
            physical_input_shape[2] = bottom_blob.h;
            physical_input_shape[3] = bottom_blob.w;
        }

        int input_axis = input_batch_axis;
        if (input_axis < 0)
            input_axis += bottom_blob.dims + 1;

        int input_shape[5] = {0, 0, 0, 0, 0};
        int input_dims = bottom_blob.dims;
        if (input_axis != 233)
        {
            if (input_axis < 0 || input_axis > bottom_blob.dims)
                return -1;

            input_dims = bottom_blob.dims + 1;
            for (int i = 0; i < input_dims; i++)
            {
                if (i < input_axis)
                    input_shape[i] = physical_input_shape[i];
                else if (i == input_axis)
                    input_shape[i] = bottom_blob.n;
                else
                    input_shape[i] = physical_input_shape[i - 1];
            }
        }
        else
        {
            if (bottom_blob.n != 1)
                return -1;

            for (int i = 0; i < input_dims; i++)
                input_shape[i] = physical_input_shape[i];
        }

        std::vector<int> output_shape;
        if (!shape_expr.empty())
        {
            int er = eval_list_expression(shape_expr, bottom_blobs, output_shape);
            if (er != 0)
                return -1;

            for (size_t i = 0; i < output_shape.size() / 2; i++)
            {
                int tmp = output_shape[i];
                output_shape[i] = output_shape[output_shape.size() - 1 - i];
                output_shape[output_shape.size() - 1 - i] = tmp;
            }
        }
        else
        {
            if (ndim == 1)
                output_shape.push_back(outw);
            if (ndim == 2)
            {
                output_shape.push_back(outh);
                output_shape.push_back(outw);
            }
            if (ndim == 3)
            {
                output_shape.push_back(outc);
                output_shape.push_back(outh);
                output_shape.push_back(outw);
            }
            if (ndim == 4)
            {
                output_shape.push_back(outc);
                output_shape.push_back(outd);
                output_shape.push_back(outh);
                output_shape.push_back(outw);
            }
        }

        const int output_dims = (int)output_shape.size();
        if (output_dims == 0 || output_dims > 5)
            return -1;

        int output_axis = output_batch_axis;
        if (output_axis < 0)
            output_axis += output_dims;

        if (output_axis != 233 && (output_axis < 0 || output_axis >= output_dims))
            return -1;

        size_t input_total = 1;
        for (int i = 0; i < input_dims; i++)
            input_total *= input_shape[i];

        size_t output_total = 1;
        int remaining_axis = -1;
        for (int i = 0; i < output_dims; i++)
        {
            if (output_shape[i] == 0)
            {
                if (i >= input_dims)
                    return -1;

                output_shape[i] = input_shape[i];
            }

            if (output_shape[i] == -1)
            {
                if (remaining_axis != -1)
                    return -1;

                remaining_axis = i;
                continue;
            }

            if (output_shape[i] <= 0)
                return -1;

            output_total *= output_shape[i];
        }

        if (remaining_axis != -1)
        {
            if (output_total == 0 || input_total % output_total != 0)
                return -1;

            output_shape[remaining_axis] = (int)(input_total / output_total);
            output_total *= output_shape[remaining_axis];
        }

        if (input_total != output_total)
            return -1;

        int batch = 1;
        int physical_output_shape[4] = {0, 0, 0, 0};
        int physical_output_dims = 0;
        for (int i = 0; i < output_dims; i++)
        {
            if (i == output_axis)
            {
                batch = output_shape[i];
                continue;
            }

            if (physical_output_dims == 4)
                return -1;

            physical_output_shape[physical_output_dims++] = output_shape[i];
        }

        if (physical_output_dims == 0)
            return -1;

        if (input_axis == output_axis && batch == bottom_blob.n)
        {
            if (physical_output_dims == 1)
                top_blob = bottom_blob.reshape(physical_output_shape[0], opt.blob_allocator);
            if (physical_output_dims == 2)
                top_blob = bottom_blob.reshape(physical_output_shape[1], physical_output_shape[0], opt.blob_allocator);
            if (physical_output_dims == 3)
                top_blob = bottom_blob.reshape(physical_output_shape[2], physical_output_shape[1], physical_output_shape[0], opt.blob_allocator);
            if (physical_output_dims == 4)
                top_blob = bottom_blob.reshape(physical_output_shape[3], physical_output_shape[2], physical_output_shape[1], physical_output_shape[0], opt.blob_allocator);

            if (top_blob.empty())
                return -100;

            return 0;
        }

        if (physical_output_dims == 1)
            top_blob.create(physical_output_shape[0], bottom_blob.elemsize, 1, batch, opt.blob_allocator);
        if (physical_output_dims == 2)
            top_blob.create(physical_output_shape[1], physical_output_shape[0], bottom_blob.elemsize, 1, batch, opt.blob_allocator);
        if (physical_output_dims == 3)
            top_blob.create(physical_output_shape[2], physical_output_shape[1], physical_output_shape[0], bottom_blob.elemsize, 1, batch, opt.blob_allocator);
        if (physical_output_dims == 4)
            top_blob.create(physical_output_shape[3], physical_output_shape[2], physical_output_shape[1], physical_output_shape[0], bottom_blob.elemsize, 1, batch, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        const unsigned char* ptr = (const unsigned char*)bottom_blob;
        unsigned char* outptr = (unsigned char*)top_blob;
        for (size_t i = 0; i < input_total;)
        {
            const size_t srcoff = get_batch_reshape_offset(bottom_blob, input_shape, input_dims, input_axis, i);
            const size_t dstoff = get_batch_reshape_offset(top_blob, &output_shape[0], output_dims, output_axis, i);

            size_t size = 1;
            while (i + size < input_total)
            {
                const size_t srcoff1 = get_batch_reshape_offset(bottom_blob, input_shape, input_dims, input_axis, i + size);
                const size_t dstoff1 = get_batch_reshape_offset(top_blob, &output_shape[0], output_dims, output_axis, i + size);
                if (srcoff1 != srcoff + size || dstoff1 != dstoff + size)
                    break;

                size++;
            }

            memcpy(outptr + dstoff * bottom_blob.elemsize, ptr + srcoff * bottom_blob.elemsize, size * bottom_blob.elemsize);

            i += size;
        }

        return 0;
    }

#endif

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c;

    int dims = bottom_blob.dims;

    if (ndim == 1)
    {
        if (outw == 0)
            outw = bottom_blob.w;

        if (outw == -1)
            outw = total;

        if (dims == 1 && bottom_blob.w == outw)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }
    if (ndim == 2)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outh == 0)
            outh = bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        if (dims == 2 && bottom_blob.h == outh)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }
    if (ndim == 3)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outh == 0)
            outh = bottom_blob.h;
        if (outc == 0)
            outc = bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outh;
        if (outh == -1)
            outh = total / outc / outw;
        if (outc == -1)
            outc = total / outh / outw;

        if (dims == 3 && bottom_blob.c == outc)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            return 0;
        }
    }
    if (ndim == 4)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outh == 0)
            outh = bottom_blob.h;
        if (outc == 0)
            outc = bottom_blob.c;
        if (outd == 0)
            outd = bottom_blob.d;

        if (outw == -1)
            outw = total / outc / outd / outh;
        if (outh == -1)
            outh = total / outc / outd / outw;
        if (outd == -1)
            outd = total / outc / outh / outw;
        if (outc == -1)
            outc = total / outd / outh / outw;

        if (dims == 4 && bottom_blob.c == outc)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }
    }


    if (ndim == 1)
    {
        top_blob = bottom_blob.reshape(outw, opt.blob_allocator);
    }
    if (ndim == 2)
    {
        top_blob = bottom_blob.reshape(outw, outh, opt.blob_allocator);
    }
    if (ndim == 3)
    {
        top_blob = bottom_blob.reshape(outw, outh, outc, opt.blob_allocator);
    }
    if (ndim == 4)
    {
        top_blob = bottom_blob.reshape(outw, outh, outd, outc, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    return 0;
}

int Reshape::eval_shape_expr(const std::vector<Mat>& bottom_blobs, int& outw, int& outh, int& outd, int& outc) const
{
    // [size(@0,0),size(@0,1),12,64]
    std::vector<int> shape;
    int er = eval_list_expression(shape_expr, bottom_blobs, shape);
    if (er != 0)
        return -1;

    outw = 1;
    outh = 1;
    outd = 1;
    outc = 1;
    if (shape.size() == 1)
    {
        outw = shape[0];
    }
    if (shape.size() == 2)
    {
        outw = shape[0];
        outh = shape[1];
    }
    if (shape.size() == 3)
    {
        outw = shape[0];
        outh = shape[1];
        outc = shape[2];
    }
    if (shape.size() == 4)
    {
        outw = shape[0];
        outh = shape[1];
        outd = shape[2];
        outc = shape[3];
    }

    return 0;
}

} // namespace ncnn
