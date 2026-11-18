// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale.h"

#include <limits.h>

namespace ncnn {

Scale::Scale()
{
    one_blob_only = true;
    support_inplace = true;
}

int Scale::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 0);
    bias_term = pd.get(1, 0);

    if (scale_data_size == -233)
        one_blob_only = false;

#if NCNN_VALIDATION
    // the scale comes from the second input and load_model reads no weights, so there is no bias_data to apply
    if (scale_data_size == -233 && bias_term)
        return -1;
#endif // NCNN_VALIDATION

    return 0;
}

int Scale::load_model(const ModelBin& mb)
{
    if (scale_data_size == -233)
        return 0;

    scale_data = mb.load(scale_data_size, 1);
    if (scale_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(scale_data_size, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

#if NCNN_VALIDATION
int Scale::validate_inputs(const std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    if (bottom_top_blobs.size() != 2)
        return -1;

    const Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];
    if (bottom_top_blob.empty() || bottom_top_blob.dims < 1 || bottom_top_blob.dims > 4
        || bottom_top_blob.w <= 0 || bottom_top_blob.h <= 0 || bottom_top_blob.d <= 0 || bottom_top_blob.c <= 0
        || bottom_top_blob.elempack <= 0 || bottom_top_blob.elempack > INT_MAX / 4)
        return -1;

    const int elempack = bottom_top_blob.elempack;
    // The kernels use int loop bounds, including packed spatial sizes.
    if (bottom_top_blob.w > INT_MAX / bottom_top_blob.h
        || bottom_top_blob.w * bottom_top_blob.h > INT_MAX / bottom_top_blob.d
        || bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d > INT_MAX / elempack)
        return -1;

    const int channels = bottom_top_blob.dims == 1 ? bottom_top_blob.w : bottom_top_blob.dims == 2 ? bottom_top_blob.h : bottom_top_blob.c;
    if (channels > INT_MAX / elempack)
        return -1;
    const int required = channels * elempack;

    const bool use_16bit = (support_fp16_storage && opt.use_fp16_storage) || (support_bf16_storage && opt.use_bf16_storage);
    const bool input_16bit = use_16bit && bottom_top_blob.elemsize == (size_t)elempack * 2;
    if (!input_16bit && bottom_top_blob.elemsize != (size_t)elempack * 4)
        return -1;

    // Scale coefficients are a contiguous vector, possibly packed or stored in 16 bits.
    if (scale_blob.empty() || scale_blob.dims != 1 || scale_blob.w <= 0
        || scale_blob.elempack <= 0 || scale_blob.elempack > INT_MAX / 4
        || scale_blob.w > INT_MAX / scale_blob.elempack
        || scale_blob.w * scale_blob.elempack < required)
        return -1;
    if (scale_blob.elemsize != (size_t)scale_blob.elempack * 4
        && !(input_16bit && scale_blob.elemsize == (size_t)scale_blob.elempack * 2))
        return -1;

    // Bias coefficients always remain fp32, including the 16-bit input paths.
    if (bias_term && (bias_data.empty() || bias_data.dims != 1 || bias_data.w <= 0
                      || bias_data.elempack <= 0 || bias_data.elempack > INT_MAX / 4
                      || bias_data.w > INT_MAX / bias_data.elempack
                      || bias_data.w * bias_data.elempack < required
                      || bias_data.elemsize != (size_t)bias_data.elempack * 4))
        return -1;

    return 0;
}
#endif // NCNN_VALIDATION

int Scale::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
#if NCNN_VALIDATION
    if (validate_inputs(bottom_top_blobs, opt) != 0)
        return -1;
#endif

    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                ptr[i] = ptr[i] * scale_blob[i] + bias_data[i];
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                ptr[i] *= scale_blob[i];
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float s = scale_blob[i];
                float bias = bias_data[i];

                for (int j = 0; j < w; j++)
                {
                    ptr[j] = ptr[j] * s + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float s = scale_blob[i];

                for (int j = 0; j < w; j++)
                {
                    ptr[j] *= s;
                }
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                float s = scale_blob[q];
                float bias = bias_data[q];

                for (int i = 0; i < size; i++)
                {
                    ptr[i] = ptr[i] * s + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                float s = scale_blob[q];

                for (int i = 0; i < size; i++)
                {
                    ptr[i] *= s;
                }
            }
        }
    }

    return 0;
}

int Scale::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_top_blobs(2);
    bottom_top_blobs[0] = bottom_top_blob;
    bottom_top_blobs[1] = scale_data;

    return forward_inplace(bottom_top_blobs, opt);
}

} // namespace ncnn
