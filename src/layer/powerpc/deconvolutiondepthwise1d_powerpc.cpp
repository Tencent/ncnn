// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deconvolutiondepthwise1d_powerpc.h"

#include "fused_activation.h"

#include <string.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

namespace ncnn {

DeconvolutionDepthWise1D_powerpc::DeconvolutionDepthWise1D_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
static NCNN_FORCEINLINE v4f32 activation_ps(v4f32 _v, int activation_type, const Mat& activation_params)
{
    if (activation_type == 0)
        return _v;

    if (activation_type == 1)
        return vec_max(_v, vec_splats(0.f));

    if (activation_type == 2)
    {
        const float slope = activation_params[0];
        v4f32 _zero = vec_splats(0.f);
        return vec_add(vec_max(_v, _zero), __ncnn_altivec_mul(vec_min(_v, _zero), vec_splats(slope)));
    }

    if (activation_type == 3)
    {
        v4f32 _min = vec_splats(activation_params[0]);
        v4f32 _max = vec_splats(activation_params[1]);
        return vec_min(vec_max(_v, _min), _max);
    }

    if (activation_type == 4)
        return sigmoid_ps(_v);

    float _t[4];
    __ncnn_altivec_store(_t, _v);
    for (int i = 0; i < 4; i++)
        _t[i] = activation_ss(_t[i], activation_type, activation_params);
    return __ncnn_altivec_load(_t);
}

// pad a pack4 1d blob along w only, the generic copy_make_border is not packing aware
static int make_padding_pack4_1d(const Mat& src, Mat& dst, int pl, int pr, float pad_value, const Option& opt)
{
    const int w = src.w;
    const int h = src.h;

    const int outw = w + pl + pr;

    dst.create(outw, h, (size_t)16u, 4, opt.workspace_allocator);
    if (dst.empty())
        return -100;

    v4f32 _pad = vec_splats(pad_value);

    for (int q = 0; q < h; q++)
    {
        const float* sptr = src.row(q);
        float* dptr = dst.row(q);

        for (int i = 0; i < outw; i++)
            __ncnn_altivec_store(dptr + i * 4, _pad);

        memcpy(dptr + (size_t)pl * 4, sptr, (size_t)w * 16);
    }

    return 0;
}
#endif // __ALTIVEC__

int DeconvolutionDepthWise1D_powerpc::forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack == 1)
        return DeconvolutionDepthWise1D::forward(bottom_blob, top_blob, opt);

    Mat bottom_blob_unpacked;
    convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
    if (bottom_blob_unpacked.empty())
        return -100;

    return DeconvolutionDepthWise1D::forward(bottom_blob_unpacked, top_blob, opt);
}

int DeconvolutionDepthWise1D_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (dynamic_weight)
        return forward_generic(bottom_blob, top_blob, opt);

    if (bottom_blob.elembits() != 32)
        return forward_generic(bottom_blob, top_blob, opt);

#if __ALTIVEC__
    // deconvolution with stride 1 dilation 1 is a flipped-tap direct conv
    // lane-wise depthwise fast path, each tap is one elementwise multiply
    const bool fast_kernel = kernel_w == 3 || kernel_w == 5;
    bool fast = fast_kernel && stride_w == 1 && dilation_w == 1
                && output_pad_right == 0 && output_w == 0
                && bottom_blob.dims == 2 && bottom_blob.elempack == 4
                && bottom_blob.h * bottom_blob.elempack == group && group == num_output;

    const int pl = pad_left;
    const int pr = pad_right;

    // the deconvolution output padding is realized as convolution input padding
    // with (kernel_extent - 1 - pad) on each side
    const int wpl = kernel_w - 1 - pl;
    const int wpr = kernel_w - 1 - pr;

    if (fast && (pl < 0 || pr < 0 || wpl < 0 || wpr < 0))
        fast = false;

    if (fast)
    {
        Mat bottom_blob_bordered;
        if (wpl > 0 || wpr > 0)
        {
            if (make_padding_pack4_1d(bottom_blob, bottom_blob_bordered, wpl, wpr, 0.f, opt) != 0)
                return -100;
        }
        else
        {
            bottom_blob_bordered = bottom_blob;
        }

        const int bw = bottom_blob_bordered.w;
        const int h = bottom_blob_bordered.h;

        const int outw = bw - kernel_w + 1;
        if (outw <= 0)
            return -100;

        const int outpack = h;

        top_blob.create(outw, outpack, (size_t)16u, 4, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* weight = weight_data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outpack; p++)
        {
            const float* kptr = weight + (size_t)p * 4 * kernel_w;

            float _wt[4];

            v4f32 _bias = vec_splats(0.f);
            if (bias_term)
                _bias = __ncnn_altivec_load((const float*)bias_data + p * 4);

            float* outptr = top_blob.row(p);

            for (int x = 0; x < outw; x++)
            {
                v4f32 _sum = _bias;

                const float* sptr = bottom_blob_bordered.row(p) + (size_t)x * 4;

                for (int k = 0; k < kernel_w; k++)
                {
                    // flip the kernel taps at runtime, weight index kernel_w - 1 - k
                    v4f32 _val = __ncnn_altivec_load(sptr + k * 4);

                    _wt[0] = kptr[0 * kernel_w + (kernel_w - 1 - k)];
                    _wt[1] = kptr[1 * kernel_w + (kernel_w - 1 - k)];
                    _wt[2] = kptr[2 * kernel_w + (kernel_w - 1 - k)];
                    _wt[3] = kptr[3 * kernel_w + (kernel_w - 1 - k)];

                    _sum = vec_madd(_val, __ncnn_altivec_load(_wt), _sum);
                }

                _sum = activation_ps(_sum, activation_type, activation_params);

                __ncnn_altivec_store(outptr + (size_t)x * 4, _sum);
            }
        }

        return 0;
    }
#endif // __ALTIVEC__

    // generic fallback, the generic kernel is not packing aware
    if (bottom_blob.elempack != 1)
    {
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        if (bottom_blob_unpacked.empty())
            return -100;

        return DeconvolutionDepthWise1D::forward(bottom_blob_unpacked, top_blob, opt);
    }

    return forward_generic(bottom_blob, top_blob, opt);
}

int DeconvolutionDepthWise1D_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // dynamic weight path, the base class kernel is not packing aware
    if (dynamic_weight)
    {
        bool any_packed = false;
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            if (bottom_blobs[i].elempack != 1)
            {
                any_packed = true;
                break;
            }
        }

        if (any_packed)
        {
            std::vector<Mat> bottom_blobs_unpacked(bottom_blobs.size());
            for (size_t i = 0; i < bottom_blobs.size(); i++)
            {
                if (bottom_blobs[i].elempack != 1)
                {
                    convert_packing(bottom_blobs[i], bottom_blobs_unpacked[i], 1, opt);
                    if (bottom_blobs_unpacked[i].empty())
                        return -100;
                }
                else
                {
                    bottom_blobs_unpacked[i] = bottom_blobs[i];
                }
            }

            return DeconvolutionDepthWise1D::forward(bottom_blobs_unpacked, top_blobs, opt);
        }
    }

    return DeconvolutionDepthWise1D::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
