// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deconvolution1d_powerpc.h"

#include "fused_activation.h"

#include <string.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

#if __PPU__
#include "gemm_powerpc_spu.h"
#endif // __PPU__

namespace ncnn {

Deconvolution1D_powerpc::Deconvolution1D_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__

    use_packed_gemm = 0;
}

int Deconvolution1D_powerpc::create_pipeline(const Option& opt)
{
    use_packed_gemm = 0;

    if (dynamic_weight)
        return 0;

#if NCNN_BF16
    if (opt.use_bf16_storage)
        return 0;
#endif

    // deconvolution with stride 1 dilation 1 is mathematically a convolution
    // with flipped weights, only the direct 1 / 3 tap kernels consume the packed layout
    if (kernel_w != 1 && kernel_w != 3)
        return 0;

    if (stride_w != 1 || dilation_w != 1)
        return 0;

    // output padding / forced output size are handled by the generic path
    if (output_pad_right != 0 || output_w != 0)
        return 0;

    const int num_input = weight_data_size / kernel_w / num_output;
    if (num_input % 4 != 0 || num_output % 4 != 0)
        return 0;

    const int inpack = num_input / 4;
    const int outpack = num_output / 4;

    // flat layout [outpack][inpack][kernel_w][4 out lane][4 in lane], kernel taps flipped
    weight_data_tm.create((size_t)inpack * outpack * 16 * kernel_w, 4u, opt.workspace_allocator);
    if (weight_data_tm.empty())
        return -100;

    const float* w = weight_data;
    float* tm = weight_data_tm;

    for (int p = 0; p < outpack; p++)
    {
        for (int q = 0; q < inpack; q++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                // flip the kernel taps once at pipeline creation
                const int kf = kernel_w - 1 - k;

                for (int j = 0; j < 4; j++)
                {
                    for (int l = 0; l < 4; l++)
                    {
                        const int o = p * 4 + j;
                        const int i = q * 4 + l;
                        tm[(((size_t)p * inpack + q) * kernel_w + k) * 16 + l * 4 + j] = w[((size_t)o * num_input + i) * kernel_w + kf];
                    }
                }
            }
        }
    }

    use_packed_gemm = 1;

    return 0;
}

int Deconvolution1D_powerpc::destroy_pipeline(const Option& /*opt*/)
{
    weight_data_tm.release();

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

int Deconvolution1D_powerpc::forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack == 1)
        return Deconvolution1D::forward(bottom_blob, top_blob, opt);

    Mat bottom_blob_unpacked;
    convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
    if (bottom_blob_unpacked.empty())
        return -100;

    return Deconvolution1D::forward(bottom_blob_unpacked, top_blob, opt);
}

int Deconvolution1D_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (dynamic_weight)
        return forward_generic(bottom_blob, top_blob, opt);

    if (bottom_blob.elembits() != 32)
        return forward_generic(bottom_blob, top_blob, opt);

#if __ALTIVEC__
    bool fast = use_packed_gemm && bottom_blob.dims == 2 && bottom_blob.elempack == 4;

    const int pl = pad_left;
    const int pr = pad_right;

    // the deconvolution output padding is realized as convolution input padding
    // with (kernel_extent - 1 - pad) on each side, the deconvolution pads must be
    // non-negative and no larger than the kernel extent to keep it representable
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
        const int inpack = bottom_blob_bordered.h;

        const int outw = bw - kernel_w + 1;
        if (outw <= 0)
            return -100;

        const int outpack = num_output / 4;

        top_blob.create(outw, outpack, (size_t)16u, 4, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* tm = weight_data_tm;

#if __PPU__
        // the spes take the aligned pack4 case: one gemm row per output
        // position, one gemm column per output channel and one K step per
        // (tap, input channel) pair. the tm taps are already flipped
        bool spe_done = false;
        if (outw % 4 == 0 && outw >= 4)
        {
            const int num_input = inpack * 4;
            const int M = outw;
            const int N = num_output;
            const int K = kernel_w * num_input;

            Mat A(K, M / 4, (size_t)16u, 4, opt.workspace_allocator);
            Mat BT(K, N / 4, (size_t)16u, 4, opt.workspace_allocator);
            Mat C(N, M / 4, (size_t)16u, 4, opt.workspace_allocator);

            if (!A.empty() && !BT.empty() && !C.empty())
            {
                // A row q, vec4 k, lane t = output position s = 4q + t at K
                // index k = tap * num_input + input channel, i.e. that input
                // channel read at position s + tap
                for (int q = 0; q < M / 4; q++)
                {
                    float* aptr = A.row(q);

                    for (int tap = 0; tap < kernel_w; tap++)
                    {
                        for (int c = 0; c < num_input; c++)
                        {
                            const size_t k = (size_t)tap * num_input + c;
                            const float* sptr = bottom_blob_bordered.row(c / 4);
                            const int cl = c % 4;

                            for (int t = 0; t < 4; t++)
                            {
                                const int s = 4 * q + t;
                                aptr[k * 4 + t] = sptr[(size_t)(s + tap) * 4 + cl];
                            }
                        }
                    }
                }

                // BT row p, vec4 k, lane j = weight of output channel 4p + j
                // at that tap / input channel. tm groups p, then q, then tap,
                // then l whereas k groups tap, then q, then l, so this is not
                // a memcpy
                for (int p = 0; p < N / 4; p++)
                {
                    float* bptr0 = BT.row(p);

                    for (int tap = 0; tap < kernel_w; tap++)
                    {
                        for (int q = 0; q < inpack; q++)
                        {
                            for (int l = 0; l < 4; l++)
                            {
                                const float* kptr = tm + (((size_t)(p * inpack + q) * kernel_w + tap) * 16) + l * 4;
                                float* bptr = bptr0 + ((size_t)tap * num_input + q * 4 + l) * 4;

                                for (int j = 0; j < 4; j++)
                                    bptr[j] = kptr[j];
                            }
                        }
                    }
                }

                if (gemm_transB_pack4_spu(A, BT, C, 1.f, 0.f, 0.f, 0) == 0)
                {
                    // C row q, vec4 j, lane t = out(position 4q + t, channel
                    // j), so top_blob.row(p) at position s wants the vec4
                    // whose lane j' is out(s, channel 4p + j')
                    for (int p = 0; p < N / 4; p++)
                    {
                        float* outptr = top_blob.row(p);

                        for (int s = 0; s < M; s++)
                        {
                            const float* cptr = C.row(s / 4);
                            const int lane = s % 4;

                            float tmp[4];
                            for (int j = 0; j < 4; j++)
                                tmp[j] = cptr[(4 * p + j) * 4 + lane];

                            v4f32 _sum = __ncnn_altivec_load(tmp);

                            if (bias_term)
                                _sum = vec_add(_sum, __ncnn_altivec_load((const float*)bias_data + p * 4));

                            _sum = activation_ps(_sum, activation_type, activation_params);

                            __ncnn_altivec_store(outptr + (size_t)s * 4, _sum);
                        }
                    }

                    spe_done = true;
                }
            }
        }
        if (!spe_done)
#endif // __PPU__
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outpack; p++)
            {
                v4f32 _bias = vec_splats(0.f);
                if (bias_term)
                    _bias = __ncnn_altivec_load((const float*)bias_data + p * 4);

                float* outptr = top_blob.row(p);

                for (int x = 0; x < outw; x++)
                {
                    v4f32 _sum = _bias;

                    for (int q = 0; q < inpack; q++)
                    {
                        const float* sptr = bottom_blob_bordered.row(q) + (size_t)x * 4;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            v4f32 _val = __ncnn_altivec_load(sptr + k * 4);
                            const float* kptr = tm + ((size_t)(p * inpack + q) * kernel_w + k) * 16;

                            _sum = vec_madd(vec_splat(_val, 0), __ncnn_altivec_load(kptr + 0), _sum);
                            _sum = vec_madd(vec_splat(_val, 1), __ncnn_altivec_load(kptr + 4), _sum);
                            _sum = vec_madd(vec_splat(_val, 2), __ncnn_altivec_load(kptr + 8), _sum);
                            _sum = vec_madd(vec_splat(_val, 3), __ncnn_altivec_load(kptr + 12), _sum);
                        }
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    __ncnn_altivec_store(outptr + (size_t)x * 4, _sum);
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

        return Deconvolution1D::forward(bottom_blob_unpacked, top_blob, opt);
    }

    return forward_generic(bottom_blob, top_blob, opt);
}

int Deconvolution1D_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

            return Deconvolution1D::forward(bottom_blobs_unpacked, top_blobs, opt);
        }
    }

    return Deconvolution1D::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
