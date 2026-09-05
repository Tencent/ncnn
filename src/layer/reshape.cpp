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

int Reshape::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

#if NCNN_BATCH
    if (input_batch_axis != 233 || output_batch_axis != 233)
        return forward_batch(bottom_blobs, top_blobs, opt);
#endif

    // resolve out shape
    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
    }

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

#if NCNN_BATCH
static size_t get_batch_reshape_offset(const Mat& m, const Mat& shape, int batch_axis, size_t i, size_t scalar_elemsize)
{
    // build logical shape
    int shape_array[5] = {0, 0, 0, 0, 0};
    int dims = shape.dims;
    {
        if (dims == 1)
            shape_array[0] = shape.w;
        if (dims == 2)
        {
            shape_array[0] = shape.h;
            shape_array[1] = shape.w;
        }
        if (dims == 3)
        {
            shape_array[0] = shape.c;
            shape_array[1] = shape.h;
            shape_array[2] = shape.w;
        }
        if (dims == 4)
        {
            shape_array[0] = shape.c;
            shape_array[1] = shape.d;
            shape_array[2] = shape.h;
            shape_array[3] = shape.w;
        }

        if (batch_axis != 233)
        {
            for (int j = dims; j > batch_axis; j--)
                shape_array[j] = shape_array[j - 1];

            shape_array[batch_axis] = shape.n;
            dims++;
        }
    }

    // linear index to logical coordinate
    int coord[5] = {0, 0, 0, 0, 0};
    {
        for (int j = dims - 1; j >= 0; j--)
        {
            coord[j] = (int)(i % shape_array[j]);
            i /= shape_array[j];
        }
    }

    // split batch coordinate from physical coordinate
    int b = 0;
    int p[4] = {0, 0, 0, 0};
    int pdims = 0;
    {
        for (int j = 0; j < dims; j++)
        {
            if (j == batch_axis)
            {
                b = coord[j];
                continue;
            }

            p[pdims++] = coord[j];
        }
    }

    // map physical coordinate to storage offset
    {
        int lane = 0;
        size_t offset = (size_t)b * m.nstep;
        if (pdims == 1)
        {
            lane = p[0] % m.elempack;
            offset += p[0] / m.elempack;
        }
        if (pdims == 2)
        {
            lane = p[0] % m.elempack;
            offset += (size_t)(p[0] / m.elempack) * m.w + p[1];
        }
        if (pdims == 3)
        {
            lane = p[0] % m.elempack;
            offset += (size_t)(p[0] / m.elempack) * m.cstep + (size_t)p[1] * m.w + p[2];
        }
        if (pdims == 4)
        {
            lane = p[0] % m.elempack;
            offset += (size_t)(p[0] / m.elempack) * m.cstep + (size_t)p[1] * m.w * m.h + (size_t)p[2] * m.w + p[3];
        }

        return offset * m.elemsize + lane * scalar_elemsize;
    }
}

int Reshape::forward_batch(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    if (bottom_blob.elempack != 1)
        return -1;

    Mat input_shape;
    Mat output_shape;
    int input_axis = 233;
    int output_axis = 233;
    size_t input_total = 0;
    if (resolve_batch_shape(bottom_blobs, input_shape, output_shape, input_axis, output_axis, input_total) != 0)
        return -1;

    if (input_axis == output_axis && output_shape.n == bottom_blob.n)
    {
        if (output_shape.dims == 1)
            top_blob = bottom_blob.reshape(output_shape.w, opt.blob_allocator);
        if (output_shape.dims == 2)
            top_blob = bottom_blob.reshape(output_shape.w, output_shape.h, opt.blob_allocator);
        if (output_shape.dims == 3)
            top_blob = bottom_blob.reshape(output_shape.w, output_shape.h, output_shape.c, opt.blob_allocator);
        if (output_shape.dims == 4)
            top_blob = bottom_blob.reshape(output_shape.w, output_shape.h, output_shape.d, output_shape.c, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        return 0;
    }

    if (output_shape.dims == 1)
        top_blob.create(output_shape.w, bottom_blob.elemsize, 1, output_shape.n, opt.blob_allocator);
    if (output_shape.dims == 2)
        top_blob.create(output_shape.w, output_shape.h, bottom_blob.elemsize, 1, output_shape.n, opt.blob_allocator);
    if (output_shape.dims == 3)
        top_blob.create(output_shape.w, output_shape.h, output_shape.c, bottom_blob.elemsize, 1, output_shape.n, opt.blob_allocator);
    if (output_shape.dims == 4)
        top_blob.create(output_shape.w, output_shape.h, output_shape.d, output_shape.c, bottom_blob.elemsize, 1, output_shape.n, opt.blob_allocator);

    if (top_blob.empty())
        return -100;

    copy_batch_reshape(bottom_blob, top_blob, input_shape, input_axis, output_shape, output_axis, input_total, bottom_blob.elemsize, opt);

    return 0;
}

void Reshape::copy_batch_reshape(const Mat& bottom_blob, Mat& top_blob,
                                 const Mat& input_shape, int input_axis,
                                 const Mat& output_shape, int output_axis,
                                 size_t total, size_t scalar_elemsize,
                                 const Option& opt) const
{
    const unsigned char* ptr = (const unsigned char*)bottom_blob;
    unsigned char* outptr = (unsigned char*)top_blob;
    const size_t block = (size_t)1 << 30;
    for (size_t i0 = 0; i0 < total; i0 += block)
    {
        const int nn = (int)(total - i0 > block ? block : total - i0);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int t = 0; t < nn; t++)
        {
            const size_t i = i0 + t;
            const size_t srcoff = get_batch_reshape_offset(bottom_blob, input_shape, input_axis, i, scalar_elemsize);
            const size_t dstoff = get_batch_reshape_offset(top_blob, output_shape, output_axis, i, scalar_elemsize);

            if (i != 0)
            {
                const size_t srcoff0 = get_batch_reshape_offset(bottom_blob, input_shape, input_axis, i - 1, scalar_elemsize);
                const size_t dstoff0 = get_batch_reshape_offset(top_blob, output_shape, output_axis, i - 1, scalar_elemsize);
                if (srcoff == srcoff0 + scalar_elemsize && dstoff == dstoff0 + scalar_elemsize)
                    continue;
            }

            size_t size = 1;
            while (i + size < total)
            {
                const size_t srcoff1 = get_batch_reshape_offset(bottom_blob, input_shape, input_axis, i + size, scalar_elemsize);
                const size_t dstoff1 = get_batch_reshape_offset(top_blob, output_shape, output_axis, i + size, scalar_elemsize);
                if (srcoff1 != srcoff + size * scalar_elemsize || dstoff1 != dstoff + size * scalar_elemsize)
                    break;

                size++;
            }

            memcpy(outptr + dstoff, ptr + srcoff, size * scalar_elemsize);
        }
    }
}

int Reshape::resolve_batch_shape(const std::vector<Mat>& bottom_blobs,
                                 Mat& input_shape, Mat& output_shape,
                                 int& input_axis, int& output_axis,
                                 size_t& total) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    input_shape = bottom_blob.shape();

    // build logical input shape
    int input_dims = input_shape.dims;
    int input_shape_array[5] = {0, 0, 0, 0, 0};
    {
        int physical_input_shape[4] = {0, 0, 0, 0};
        if (input_shape.dims == 1)
            physical_input_shape[0] = input_shape.w;
        if (input_shape.dims == 2)
        {
            physical_input_shape[0] = input_shape.h;
            physical_input_shape[1] = input_shape.w;
        }
        if (input_shape.dims == 3)
        {
            physical_input_shape[0] = input_shape.c;
            physical_input_shape[1] = input_shape.h;
            physical_input_shape[2] = input_shape.w;
        }
        if (input_shape.dims == 4)
        {
            physical_input_shape[0] = input_shape.c;
            physical_input_shape[1] = input_shape.d;
            physical_input_shape[2] = input_shape.h;
            physical_input_shape[3] = input_shape.w;
        }

        input_axis = input_batch_axis;
        if (input_axis < 0)
            input_axis += input_shape.dims + 1;

        if (input_axis != 233)
        {
            if (input_axis < 0 || input_axis > input_shape.dims)
                return -1;

            input_dims = input_shape.dims + 1;
            for (int i = 0; i < input_dims; i++)
            {
                if (i < input_axis)
                    input_shape_array[i] = physical_input_shape[i];
                else if (i == input_axis)
                    input_shape_array[i] = input_shape.n;
                else
                    input_shape_array[i] = physical_input_shape[i - 1];
            }
        }
        else
        {
            if (input_shape.n != 1)
                return -1;

            for (int i = 0; i < input_dims; i++)
                input_shape_array[i] = physical_input_shape[i];
        }
    }

    // resolve output shape
    std::vector<int> outshape;
    {
        if (!shape_expr.empty())
        {
            int er = eval_list_expression(shape_expr, bottom_blobs, outshape);
            if (er != 0)
                return -1;

            for (size_t i = 0; i < outshape.size() / 2; i++)
            {
                int tmp = outshape[i];
                outshape[i] = outshape[outshape.size() - 1 - i];
                outshape[outshape.size() - 1 - i] = tmp;
            }
        }
        else
        {
            if (ndim == 1)
                outshape.push_back(w);
            if (ndim == 2)
            {
                outshape.push_back(h);
                outshape.push_back(w);
            }
            if (ndim == 3)
            {
                outshape.push_back(c);
                outshape.push_back(h);
                outshape.push_back(w);
            }
            if (ndim == 4)
            {
                outshape.push_back(c);
                outshape.push_back(d);
                outshape.push_back(h);
                outshape.push_back(w);
            }
        }
    }

    int output_dims = (int)outshape.size();
    if (output_dims == 0 || output_dims > 5)
        return -1;

    output_axis = output_batch_axis;
    if (output_axis < 0)
        output_axis += output_dims;

    if (output_axis != 233 && (output_axis < 0 || output_axis >= output_dims))
        return -1;

    // materialize output shape
    int output_shape_array[5] = {0, 0, 0, 0, 0};
    {
        total = 1;
        for (int i = 0; i < input_dims; i++)
            total *= input_shape_array[i];

        size_t output_total = 1;
        int remaining_axis = -1;
        for (int i = 0; i < output_dims; i++)
        {
            output_shape_array[i] = outshape[i];

            if (output_shape_array[i] == 0)
            {
                if (i >= input_dims)
                    return -1;

                output_shape_array[i] = input_shape_array[i];
            }

            if (output_shape_array[i] == -1)
            {
                if (remaining_axis != -1)
                    return -1;

                remaining_axis = i;
                continue;
            }

            if (output_shape_array[i] <= 0)
                return -1;

            output_total *= output_shape_array[i];
        }

        if (remaining_axis != -1)
        {
            if (output_total == 0 || total % output_total != 0)
                return -1;

            output_shape_array[remaining_axis] = (int)(total / output_total);
            output_total *= output_shape_array[remaining_axis];
        }

        if (total != output_total)
            return -1;
    }

    // build physical output shape
    {
        int n = 1;
        int physical_output_dims = 0;
        int physical_output_shape[4] = {0, 0, 0, 0};
        for (int i = 0; i < output_dims; i++)
        {
            if (i == output_axis)
            {
                n = output_shape_array[i];
                continue;
            }

            if (physical_output_dims == 4)
                return -1;

            physical_output_shape[physical_output_dims++] = output_shape_array[i];
        }

        if (physical_output_dims == 0)
            return -1;

        if (physical_output_dims == 1)
            output_shape = Mat(physical_output_shape[0], (void*)0, 4u, 1, n);
        if (physical_output_dims == 2)
            output_shape = Mat(physical_output_shape[1], physical_output_shape[0], (void*)0, 4u, 1, n);
        if (physical_output_dims == 3)
            output_shape = Mat(physical_output_shape[2], physical_output_shape[1], physical_output_shape[0], (void*)0, 4u, 1, n);
        if (physical_output_dims == 4)
            output_shape = Mat(physical_output_shape[3], physical_output_shape[2], physical_output_shape[1], physical_output_shape[0], (void*)0, 4u, 1, n);
    }

    return 0;
}

#endif // NCNN_BATCH

} // namespace ncnn
