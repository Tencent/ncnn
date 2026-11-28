// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eltwise.h"

namespace ncnn {

Eltwise::Eltwise()
{
    one_blob_only = false;
    // stays false for the graph machinery: net.cpp dispatches Vulkan and
    // light-mode blob reuse by this flag, and neither has an in-place
    // Eltwise path (Eltwise_vulkan has no forward_inplace; the shared
    // test infrastructure allocates single-output vectors). Direct layer
    // callers may still invoke forward_inplace below explicitly.
    support_inplace = false;
}

int Eltwise::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    coeffs = pd.get(1, Mat());

    const int coeffs_type = pd.type(1);

#if NCNN_VALIDATION
    if (op_type < Operation_PROD || op_type > Operation_MAX)
        return -1;

    if (coeffs_type != 0 && coeffs_type != 4 && coeffs_type != 5 && coeffs_type != 6)
        return -1;

    if ((coeffs.dims != 0 || coeffs.w != 0 || coeffs.data) && (coeffs.dims != 1 || coeffs.w < 0 || coeffs.elempack != 1 || coeffs.elemsize != 4u || (coeffs.w > 0 && !coeffs.data)))
        return -1;
#endif // NCNN_VALIDATION

    // convert integer text arrays without modifying the shared data
    if (coeffs_type == 5 && !coeffs.empty())
    {
        Mat converted(coeffs.w);
        if (converted.empty())
            return -100;

        const int* p = coeffs;
        for (int i = 0; i < coeffs.w; i++)
            converted[i] = (float)p[i];

        coeffs = converted;
    }

    return 0;
}

int Eltwise::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int size = w * h * d;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = ptr[i] * ptr1[i];
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] *= ptr[i];
                }
            }
        }
    }
    else if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] + ptr1[i];
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i];
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] * coeff0 + ptr1[i] * coeff1;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i] * coeff;
                    }
                }
            }
        }
    }
    else if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = std::max(ptr[i], ptr1[i]);
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = std::max(outptr[i], ptr[i]);
                }
            }
        }
    }

    return 0;
}

int Eltwise::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    // fold every other blob into the first one in place: SUM is
    // coeffs[0]*b0 + sum coeffs[b]*b_b (any coeffs), PROD is b0 * prod b_b,
    // MAX is max(b0, b_1..). float storage only: 16-bit storage layouts
    // (fp16/bf16) have their own out-of-place kernels, and a float-pointer
    // fold here would reinterpret pairs of half elements. check the
    // per-scalar bit width, not the per-channel byte size: a packed
    // tensor's elemsize is scaled by elempack (pack4 fp32 = 16 bytes,
    // pack2 fp16 = 4 bytes)
    Mat& bottom_blob = bottom_top_blobs[0];
    if (bottom_blob.elembits() != 32)
        return -1;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    // packed layouts store elempack scalars per channel cell
    int size = w * h * d * bottom_blob.elempack;

    if (op_type == Operation_SUM)
    {
        // apply the leading coefficient in place first: coeffs[0] may be
        // any value (ncnnoptimize writes arbitrary BinaryOp scales there)
        if (coeffs.w != 0 && coeffs[0] != 1.f)
        {
            const float c0 = coeffs[0];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = bottom_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] *= c0;
                }
            }
        }

        for (size_t b = 1; b < bottom_top_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_top_blobs[b];
            const float coeff = coeffs.w == 0 ? 1.f : coeffs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = bottom_blob.channel(q);
                const float* ptr = bottom_blob1.channel(q);

                if (coeff == 1.f)
                {
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i];
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i] * coeff;
                    }
                }
            }
        }

        return 0;
    }

    if (op_type == Operation_PROD)
    {
        for (size_t b = 1; b < bottom_top_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_top_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = bottom_blob.channel(q);
                const float* ptr = bottom_blob1.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] *= ptr[i];
                }
            }
        }

        return 0;
    }

    if (op_type == Operation_MAX)
    {
        for (size_t b = 1; b < bottom_top_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_top_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = bottom_blob.channel(q);
                const float* ptr = bottom_blob1.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = std::max(outptr[i], ptr[i]);
                }
            }
        }

        return 0;
    }

    return -1;
}

} // namespace ncnn
