// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deconvolutiondepthwise3d_powerpc.h"

#include "fused_activation.h"

#include <string.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

namespace ncnn {

DeconvolutionDepthWise3D_powerpc::DeconvolutionDepthWise3D_powerpc()
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

// pad a pack4 3d blob, the generic copy_make_border_3d is not packing aware
static int make_padding_pack4_3d(const Mat& src, Mat& dst, int pt, int pb, int pl, int pr, int pf, int pbk, float pad_value, const Option& opt)
{
    const int w = src.w;
    const int h = src.h;
    const int d = src.d;
    const int c = src.c;

    const int outw = w + pl + pr;
    const int outh = h + pt + pb;
    const int outd = d + pf + pbk;

    dst.create(outw, outh, outd, c, (size_t)16u, 4, opt.workspace_allocator);
    if (dst.empty())
        return -100;

    v4f32 _pad = vec_splats(pad_value);

    for (int q = 0; q < c; q++)
    {
        const float* sptr = src.channel(q);
        float* dptr = dst.channel(q);

        for (size_t i = 0; i < (size_t)outw * outh * outd; i++)
            __ncnn_altivec_store(dptr + i * 4, _pad);

        for (int z = 0; z < d; z++)
        {
            for (int y = 0; y < h; y++)
            {
                memcpy(dptr + (((size_t)(z + pf) * outh + (y + pt)) * outw + pl) * 4, sptr + ((size_t)z * h + y) * w * 4, (size_t)w * 16);
            }
        }
    }

    return 0;
}
#endif // __ALTIVEC__

int DeconvolutionDepthWise3D_powerpc::forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack == 1)
        return DeconvolutionDepthWise3D::forward(bottom_blob, top_blob, opt);

    Mat bottom_blob_unpacked;
    convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
    if (bottom_blob_unpacked.empty())
        return -100;

    return DeconvolutionDepthWise3D::forward(bottom_blob_unpacked, top_blob, opt);
}

int DeconvolutionDepthWise3D_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elembits() != 32)
        return forward_generic(bottom_blob, top_blob, opt);

#if __ALTIVEC__
    // deconvolution with stride 1 dilation 1 is a flipped-tap direct conv
    // lane-wise depthwise fast path, each tap is one elementwise multiply
    const bool fast_kernel = (kernel_w == 3 && kernel_h == 3 && kernel_d == 3) || (kernel_w == 5 && kernel_h == 5 && kernel_d == 5);
    bool fast = fast_kernel
                && stride_w == 1 && stride_h == 1 && stride_d == 1
                && dilation_w == 1 && dilation_h == 1 && dilation_d == 1
                && output_pad_right == 0 && output_pad_bottom == 0 && output_pad_behind == 0
                && output_w == 0 && output_h == 0 && output_d == 0
                && bottom_blob.dims == 4 && bottom_blob.elempack == 4
                && bottom_blob.c * bottom_blob.elempack == group && group == num_output;

    const int pt = pad_top;
    const int pb = pad_bottom;
    const int pl = pad_left;
    const int pr = pad_right;
    const int pf = pad_front;
    const int pbk = pad_behind;

    // the deconvolution output padding is realized as convolution input padding
    // with (kernel_extent - 1 - pad) on each side
    const int wpt = kernel_h - 1 - pt;
    const int wpb = kernel_h - 1 - pb;
    const int wpl = kernel_w - 1 - pl;
    const int wpr = kernel_w - 1 - pr;
    const int wpf = kernel_d - 1 - pf;
    const int wpbk = kernel_d - 1 - pbk;

    if (fast && (pt < 0 || pb < 0 || pl < 0 || pr < 0 || pf < 0 || pbk < 0
                 || wpt < 0 || wpb < 0 || wpl < 0 || wpr < 0 || wpf < 0 || wpbk < 0))
        fast = false;

    if (fast)
    {
        Mat bottom_blob_bordered;
        if (wpt > 0 || wpb > 0 || wpl > 0 || wpr > 0 || wpf > 0 || wpbk > 0)
        {
            if (make_padding_pack4_3d(bottom_blob, bottom_blob_bordered, wpt, wpb, wpl, wpr, wpf, wpbk, 0.f, opt) != 0)
                return -100;
        }
        else
        {
            bottom_blob_bordered = bottom_blob;
        }

        const int bw = bottom_blob_bordered.w;
        const int bh = bottom_blob_bordered.h;
        const int bd = bottom_blob_bordered.d;
        const int c = bottom_blob_bordered.c;

        const int outw = bw - kernel_w + 1;
        const int outh = bh - kernel_h + 1;
        const int outd = bd - kernel_d + 1;
        if (outw <= 0 || outh <= 0 || outd <= 0)
            return -100;

        const int maxk = kernel_w * kernel_h * kernel_d;

        top_blob.create(outw, outh, outd, c, (size_t)16u, 4, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* weight = weight_data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < c; p++)
        {
            const float* base_w = weight + (size_t)p * 4 * maxk;

            float _wt[4];

            v4f32 _bias = vec_splats(0.f);
            if (bias_term)
                _bias = __ncnn_altivec_load((const float*)bias_data + p * 4);

            const float* inptr = bottom_blob_bordered.channel(p);

            float* outptr = top_blob.channel(p);

            for (int z = 0; z < outd; z++)
            {
                for (int y = 0; y < outh; y++)
                {
                    for (int x = 0; x < outw; x++)
                    {
                        v4f32 _sum = _bias;

                        for (int kd = 0; kd < kernel_d; kd++)
                        {
                            for (int ky = 0; ky < kernel_h; ky++)
                            {
                                const float* sptr = inptr + (((size_t)(z + kd) * bh + (y + ky)) * bw + x) * 4;

                                for (int kx = 0; kx < kernel_w; kx++)
                                {
                                    const int k = (kd * kernel_h + ky) * kernel_w + kx;

                                    // flip the kernel taps at runtime, weight index maxk - 1 - k
                                    v4f32 _val = __ncnn_altivec_load(sptr + kx * 4);

                                    _wt[0] = base_w[0 * maxk + (maxk - 1 - k)];
                                    _wt[1] = base_w[1 * maxk + (maxk - 1 - k)];
                                    _wt[2] = base_w[2 * maxk + (maxk - 1 - k)];
                                    _wt[3] = base_w[3 * maxk + (maxk - 1 - k)];

                                    _sum = vec_madd(_val, __ncnn_altivec_load(_wt), _sum);
                                }
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        __ncnn_altivec_store(outptr, _sum);
                        outptr += 4;
                    }
                }
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

        return DeconvolutionDepthWise3D::forward(bottom_blob_unpacked, top_blob, opt);
    }

    return forward_generic(bottom_blob, top_blob, opt);
}

} // namespace ncnn
