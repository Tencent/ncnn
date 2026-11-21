// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deconvolutiondepthwise_powerpc.h"

#include "fused_activation.h"

#include <string.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

namespace ncnn {

DeconvolutionDepthWise_powerpc::DeconvolutionDepthWise_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__

    use_packed_dw = 0;
}

int DeconvolutionDepthWise_powerpc::create_pipeline(const Option& opt)
{
    use_packed_dw = 0;

    if (dynamic_weight)
        return 0;

#if NCNN_BF16
    if (opt.use_bf16_storage)
        return 0;
#endif

    const int maxk = kernel_w * kernel_h;
    const int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise only
    if (channels != group || group != num_output)
        return 0;

    // deconvolution with stride 1 dilation 1 is mathematically a convolution
    // with spatially flipped weights
    const bool fast_kernel = (kernel_w == 3 && kernel_h == 3) || (kernel_w == 5 && kernel_h == 5);
    if (!fast_kernel || stride_w != 1 || stride_h != 1 || dilation_w != 1 || dilation_h != 1)
        return 0;

    // output padding / forced output size are handled by the generic path
    if (output_pad_right != 0 || output_pad_bottom != 0 || output_w != 0 || output_h != 0)
        return 0;

    if (channels % 4 != 0)
        return 0;

    // flat layout [maxk][channels/4][4 channel lane], kernel taps flipped
    weight_data_tm.create((size_t)maxk * channels, 4u, opt.workspace_allocator);
    if (weight_data_tm.empty())
        return -100;

    const float* w = weight_data;
    float* tm = weight_data_tm;

    for (int gp = 0; gp < channels / 4; gp++)
    {
        for (int l = 0; l < 4; l++)
        {
            const int g = gp * 4 + l;
            for (int k = 0; k < maxk; k++)
            {
                // flip the kernel taps once at pipeline creation
                tm[(gp * maxk + k) * 4 + l] = w[g * maxk + (maxk - 1 - k)];
            }
        }
    }

    use_packed_dw = 1;

    return 0;
}

int DeconvolutionDepthWise_powerpc::destroy_pipeline(const Option& /*opt*/)
{
    weight_data_tm.release();

    return 0;
}

#if __ALTIVEC__
static NCNN_FORCEINLINE v4f32 deconvdw_activation_ps(v4f32 _v, int activation_type, const Mat& activation_params)
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

// pad a pack4 blob, the generic copy_make_border is not packing aware
static int deconvdw_make_padding_pack4(const Mat& src, Mat& dst, int pt, int pb, int pl, int pr, float pad_value, const Option& opt)
{
    const int w = src.w;
    const int h = src.h;
    const int c = src.c;

    const int outw = w + pl + pr;
    const int outh = h + pt + pb;

    dst.create(outw, outh, c, (size_t)16u, 4, opt.workspace_allocator);
    if (dst.empty())
        return -100;

    v4f32 _pad = vec_splats(pad_value);

    for (int q = 0; q < c; q++)
    {
        const float* sptr = src.channel(q);
        float* dptr = dst.channel(q);

        for (int i = 0; i < outw * outh; i++)
            __ncnn_altivec_store(dptr + i * 4, _pad);

        for (int y = 0; y < h; y++)
            memcpy(dptr + ((size_t)(y + pt) * outw + pl) * 4, sptr + (size_t)y * w * 4, (size_t)w * 16);
    }

    return 0;
}
#endif // __ALTIVEC__

int DeconvolutionDepthWise_powerpc::forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack == 1)
        return DeconvolutionDepthWise::forward(bottom_blob, top_blob, opt);

    Mat bottom_blob_unpacked;
    convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
    if (bottom_blob_unpacked.empty())
        return -100;

    return DeconvolutionDepthWise::forward(bottom_blob_unpacked, top_blob, opt);
}

int DeconvolutionDepthWise_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (dynamic_weight)
        return forward_generic(bottom_blob, top_blob, opt);

    if (bottom_blob.elembits() != 32)
        return forward_generic(bottom_blob, top_blob, opt);

#if __ALTIVEC__
    const int channels = bottom_blob.c * bottom_blob.elempack;

    bool fast = use_packed_dw
                && bottom_blob.dims == 3
                && bottom_blob.elempack == 4
                && num_output % 4 == 0
                && channels == group && group == num_output;

    const int pt = pad_top;
    const int pb = pad_bottom;
    const int pl = pad_left;
    const int pr = pad_right;

    // the deconvolution output padding is realized as convolution input padding
    // with (kernel_extent - 1 - pad) on each side, the deconvolution pads must be
    // non-negative and no larger than the kernel extent to keep it representable
    const int wpt = kernel_h - 1 - pt;
    const int wpb = kernel_h - 1 - pb;
    const int wpl = kernel_w - 1 - pl;
    const int wpr = kernel_w - 1 - pr;

    if (fast && (pt < 0 || pb < 0 || pl < 0 || pr < 0 || wpt < 0 || wpb < 0 || wpl < 0 || wpr < 0))
        fast = false;

    if (fast)
    {
        Mat bottom_blob_bordered;
        if (deconvdw_make_padding_pack4(bottom_blob, bottom_blob_bordered, wpt, wpb, wpl, wpr, 0.f, opt) != 0)
            return -100;

        const int bw = bottom_blob_bordered.w;
        const int bh = bottom_blob_bordered.h;

        const int outw = bw - kernel_w + 1;
        const int outh = bh - kernel_h + 1;
        if (outw <= 0 || outh <= 0)
            return -100;

        // with pack4 the 4 lanes are independent channels, each tap is a single vec_madd
        const int gpack = bottom_blob_bordered.c;
        const int maxk = kernel_w * kernel_h;

        top_blob.create(outw, outh, gpack, (size_t)16u, 4, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* tm = weight_data_tm;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < gpack; g++)
        {
            const float* kptr = tm + (size_t)g * maxk * 4;
            const float* inptr = bottom_blob_bordered.channel(g);
            float* outptr = top_blob.channel(g);

            v4f32 _bias = vec_splats(0.f);
            if (bias_term)
                _bias = __ncnn_altivec_load((const float*)bias_data + g * 4);

            for (int y = 0; y < outh; y++)
            {
                for (int x = 0; x < outw; x++)
                {
                    v4f32 _sum = _bias;

                    for (int ky = 0; ky < kernel_h; ky++)
                    {
                        const float* sptr = inptr + ((size_t)(y + ky) * bw + x) * 4;
                        const float* krow = kptr + ky * kernel_w * 4;

                        for (int kx = 0; kx < kernel_w; kx++)
                        {
                            v4f32 _val = __ncnn_altivec_load(sptr + kx * 4);
                            v4f32 _w = __ncnn_altivec_load(krow + kx * 4);

                            _sum = vec_madd(_val, _w, _sum);
                        }
                    }

                    _sum = deconvdw_activation_ps(_sum, activation_type, activation_params);

                    __ncnn_altivec_store(outptr, _sum);
                    outptr += 4;
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

        return DeconvolutionDepthWise::forward(bottom_blob_unpacked, top_blob, opt);
    }

    return forward_generic(bottom_blob, top_blob, opt);
}

int DeconvolutionDepthWise_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // the generic multi-blob (dynamic weight) kernel is not packing aware
    bool any_packed = false;
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (bottom_blobs[i].elempack != 1)
        {
            any_packed = true;
            break;
        }
    }

    if (!any_packed)
        return DeconvolutionDepthWise::forward(bottom_blobs, top_blobs, opt);

    std::vector<Mat> bottom_blobs_unpacked(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (bottom_blobs[i].elempack == 1)
        {
            bottom_blobs_unpacked[i] = bottom_blobs[i];
            continue;
        }

        convert_packing(bottom_blobs[i], bottom_blobs_unpacked[i], 1, opt);
        if (bottom_blobs_unpacked[i].empty())
            return -100;
    }

    return DeconvolutionDepthWise::forward(bottom_blobs_unpacked, top_blobs, opt);
}

} // namespace ncnn
