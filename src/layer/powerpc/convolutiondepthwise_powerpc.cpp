// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolutiondepthwise_powerpc.h"

#include "fused_activation.h"

#include <string.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

namespace ncnn {

ConvolutionDepthWise_powerpc::ConvolutionDepthWise_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__

    activation = 0;
    use_packed_weights = 0;
}

int ConvolutionDepthWise_powerpc::create_pipeline(const Option& opt)
{
    use_packed_weights = 0;

    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
        return 0;
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage)
        return 0;
#endif

    const int maxk = kernel_w * kernel_h;
    const int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels != group || group != num_output)
        return 0;

#if __ALTIVEC__
    // only the direct 3x3 / 5x5 stride1/stride2 dilation1 kernels consume the packed layout
    const bool fast_kernel = (kernel_w == 3 && kernel_h == 3) || (kernel_w == 5 && kernel_h == 5);
    if (!fast_kernel || dilation_w != 1 || dilation_h != 1 || (stride_w != 1 && stride_w != 2) || (stride_h != 1 && stride_h != 2) || stride_w != stride_h)
        return 0;

    if (channels % 4 != 0)
        return 0;

    // pack4, each weight element is a 4-float vector of 4 channels
    Mat weight_data_r2 = weight_data.reshape(maxk, group);
    convert_packing(weight_data_r2, weight_data_tm, 4, opt);
    if (weight_data_tm.empty())
        return -100;

    use_packed_weights = 1;
#endif // __ALTIVEC__

    return 0;
}

int ConvolutionDepthWise_powerpc::destroy_pipeline(const Option& /*opt*/)
{
    if (activation)
    {
        delete activation;
        activation = 0;
    }

    weight_data_tm.release();
    use_packed_weights = 0;

    return 0;
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

// pad a pack4 blob, the generic copy_make_border is not packing aware
static int make_padding_pack4(const Mat& src, Mat& dst, int pt, int pb, int pl, int pr, float pad_value, const Option& opt)
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

int ConvolutionDepthWise_powerpc::forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack == 1)
        return ConvolutionDepthWise::forward(bottom_blob, top_blob, opt);

    Mat bottom_blob_unpacked;
    convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
    if (bottom_blob_unpacked.empty())
        return -100;

    return ConvolutionDepthWise::forward(bottom_blob_unpacked, top_blob, opt);
}

int ConvolutionDepthWise_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
        return forward_generic(bottom_blob, top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blob.elembits() == 16)
        return forward_generic(bottom_blob, top_blob, opt);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int out_elempack = 1;
#if __ALTIVEC__
    if (opt.use_packing_layout)
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif

    bool fast = false;

#if __ALTIVEC__
    if (use_packed_weights && !dynamic_weight && bottom_blob.dims == 3 && bottom_blob.elembits() == 32
            && elempack == 4 && out_elempack == 4
            && channels * elempack == group && group == num_output)
    {
        fast = true;

        int pt = pad_top;
        int pb = pad_bottom;
        int pl = pad_left;
        int pr = pad_right;

        if (pl == -233 && pr == -233 && pt == -233 && pb == -233)
        {
            // tensorflow padding=SAME or onnx padding=SAME_UPPER
            const int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
            const int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
            pl = wpad / 2;
            pr = wpad - pl;
            pt = hpad / 2;
            pb = hpad - pt;
        }
        else if (pl == -234 && pr == -234 && pt == -234 && pb == -234)
        {
            // onnx padding=SAME_LOWER
            const int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
            const int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
            pl = wpad - wpad / 2;
            pr = wpad / 2;
            pt = hpad - hpad / 2;
            pb = hpad / 2;
        }
        else if (pl < 0 || pr < 0 || pt < 0 || pb < 0)
        {
            fast = false;
        }

        if (fast)
        {
            Mat bottom_blob_bordered;
            if (pt > 0 || pb > 0 || pl > 0 || pr > 0)
            {
                if (make_padding_pack4(bottom_blob, bottom_blob_bordered, pt, pb, pl, pr, pad_value, opt) != 0)
                    return -100;
            }
            else
            {
                bottom_blob_bordered = bottom_blob;
            }

            w = bottom_blob_bordered.w;
            h = bottom_blob_bordered.h;

            const int outw = (w - kernel_extent_w) / stride_w + 1;
            const int outh = (h - kernel_extent_h) / stride_h + 1;
            if (outw <= 0 || outh <= 0)
                return -100;

            top_blob.create(outw, outh, num_output / 4, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const int maxk = kernel_w * kernel_h;

            // kernel offsets
            std::vector<int> _space_ofs(maxk);
            int* space_ofs = &_space_ofs[0];
            {
                int p1 = 0;
                int p2 = 0;
                int gap = w * dilation_h - kernel_w * dilation_w;
                for (int i = 0; i < kernel_h; i++)
                {
                    for (int j = 0; j < kernel_w; j++)
                    {
                        space_ofs[p1] = p2;
                        p1++;
                        p2 += dilation_w;
                    }
                    p2 += gap;
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                float* outptr = top_blob.channel(g);
                const float* kptr = (const float*)weight_data_tm + maxk * g * 4;
                const Mat m = bottom_blob_bordered.channel(g);

                v4f32 _bias = vec_splats(0.f);
                if (bias_term)
                    _bias = __ncnn_altivec_load((const float*)bias_data + g * 4);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        v4f32 _sum = _bias;

                        const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                        for (int k = 0; k < maxk; k++)
                        {
                            v4f32 _val = __ncnn_altivec_load(sptr + space_ofs[k] * 4);
                            v4f32 _w = __ncnn_altivec_load(kptr + k * 4);
                            _sum = vec_madd(_val, _w, _sum);
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        __ncnn_altivec_store(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }

            return 0;
        }
    }
#endif // __ALTIVEC__

    (void)w;
    (void)h;
    (void)channels;
    (void)elemsize;
    (void)elempack;
    (void)out_elempack;
    (void)kernel_extent_w;
    (void)kernel_extent_h;

    // generic fallback, the generic kernel is not packing aware
    if (bottom_blob.elempack != 1)
    {
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        if (bottom_blob_unpacked.empty())
            return -100;

        return ConvolutionDepthWise::forward(bottom_blob_unpacked, top_blob, opt);
    }

    return forward_generic(bottom_blob, top_blob, opt);
}

int ConvolutionDepthWise_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
        return ConvolutionDepthWise::forward(bottom_blobs, top_blobs, opt);

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

    return ConvolutionDepthWise::forward(bottom_blobs_unpacked, top_blobs, opt);
}

} // namespace ncnn
