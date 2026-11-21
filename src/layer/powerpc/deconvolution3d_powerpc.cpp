// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deconvolution3d_powerpc.h"

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

Deconvolution3D_powerpc::Deconvolution3D_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__

    use_packed_gemm = 0;
}

int Deconvolution3D_powerpc::create_pipeline(const Option& opt)
{
    use_packed_gemm = 0;

#if NCNN_BF16
    if (opt.use_bf16_storage)
        return 0;
#endif

    const int maxk = kernel_w * kernel_h * kernel_d;

    // deconvolution with stride 1 dilation 1 is mathematically a convolution
    // with spatially flipped weights, only the direct 1x1x1 and 3x3x3 kernels
    // consume the packed layout
    const bool fast_kernel = (kernel_w == 1 && kernel_h == 1 && kernel_d == 1) || (kernel_w == 3 && kernel_h == 3 && kernel_d == 3);
    if (!fast_kernel || stride_w != 1 || stride_h != 1 || stride_d != 1 || dilation_w != 1 || dilation_h != 1 || dilation_d != 1)
        return 0;

    // output padding / forced output size are handled by the generic path
    if (output_pad_right != 0 || output_pad_bottom != 0 || output_pad_behind != 0 || output_w != 0 || output_h != 0 || output_d != 0)
        return 0;

    const int num_input = weight_data_size / maxk / num_output;
    if (num_input % 4 != 0 || num_output % 4 != 0)
        return 0;

    const int inpack = num_input / 4;
    const int outpack = num_output / 4;
    const int blocksize = inpack * outpack * 16;

    // flat layout [maxk][inpack][outpack][4 in lane][4 out lane], kernel taps flipped
    weight_data_tm.create((size_t)blocksize * maxk, 4u, opt.workspace_allocator);
    if (weight_data_tm.empty())
        return -100;

    const float* w = weight_data;
    float* tm = weight_data_tm;

    for (int k = 0; k < maxk; k++)
    {
        float* gk = tm + (size_t)k * blocksize;

        // flip the kernel taps once at pipeline creation
        const int kf = maxk - 1 - k;

        for (int q = 0; q < inpack; q++)
        {
            for (int p = 0; p < outpack; p++)
            {
                for (int l = 0; l < 4; l++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const int o = p * 4 + j;
                        const int i = q * 4 + l;
                        gk[((q * outpack + p) * 4 + l) * 4 + j] = w[((size_t)o * num_input + i) * maxk + kf];
                    }
                }
            }
        }
    }

    use_packed_gemm = 1;

    return 0;
}

int Deconvolution3D_powerpc::destroy_pipeline(const Option& /*opt*/)
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

int Deconvolution3D_powerpc::forward_generic(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack == 1)
        return Deconvolution3D::forward(bottom_blob, top_blob, opt);

    Mat bottom_blob_unpacked;
    convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
    if (bottom_blob_unpacked.empty())
        return -100;

    return Deconvolution3D::forward(bottom_blob_unpacked, top_blob, opt);
}

int Deconvolution3D_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elembits() != 32)
        return forward_generic(bottom_blob, top_blob, opt);

#if __ALTIVEC__
    bool fast = use_packed_gemm && bottom_blob.dims == 4 && bottom_blob.elempack == 4;

    const int pt = pad_top;
    const int pb = pad_bottom;
    const int pl = pad_left;
    const int pr = pad_right;
    const int pf = pad_front;
    const int pbk = pad_behind;

    // the deconvolution output padding is realized as convolution input padding
    // with (kernel_extent - 1 - pad) on each side, the deconvolution pads must be
    // non-negative and no larger than the kernel extent to keep it representable
    const int wpt = kernel_h - 1 - pt;
    const int wpb = kernel_h - 1 - pb;
    const int wpl = kernel_w - 1 - pl;
    const int wpr = kernel_w - 1 - pr;
    const int wpf = kernel_d - 1 - pf;
    const int wpbk = kernel_d - 1 - pbk;

    if (fast && (pt < 0 || pb < 0 || pl < 0 || pr < 0 || pf < 0 || pbk < 0 || wpt < 0 || wpb < 0 || wpl < 0 || wpr < 0 || wpf < 0 || wpbk < 0))
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
        const int inpack = bottom_blob_bordered.c;

        const int outw = bw - kernel_w + 1;
        const int outh = bh - kernel_h + 1;
        const int outd = bd - kernel_d + 1;
        if (outw <= 0 || outh <= 0 || outd <= 0)
            return -100;

        const int outpack = num_output / 4;
        const int blocksize = inpack * outpack * 16;

        top_blob.create(outw, outh, outd, outpack, (size_t)16u, 4, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* tm = weight_data_tm;

#if __PPU__
        // the spes take the aligned pack4 case; anything else, or a spe that
        // fails to start, falls through to the ppe kernel
        bool spe_done = false;

        const int S = outw * outh * outd;
        const int num_input = inpack * 4;
        const int maxk = kernel_w * kernel_h * kernel_d;
        const int K = maxk * num_input;

        if (S % 4 == 0 && S >= 4)
        {
            Mat A(K, S / 4, (size_t)16u, 4, opt.workspace_allocator);
            Mat BT(K, outpack, (size_t)16u, 4, opt.workspace_allocator);
            Mat C(num_output, S / 4, (size_t)16u, 4, opt.workspace_allocator);

            if (!A.empty() && !BT.empty() && !C.empty())
            {
                // the im2col gather: k index k_tap*num_input+c is input channel
                // c of kernel tap k_tap, and the 4 lanes of an A vec4 are the 4
                // output voxels 4q..4q+3 reading that one channel. the taps in
                // tm are already flipped at pipeline creation
                for (int q = 0; q < S / 4; q++)
                {
                    float* aptr = A.row(q);

                    for (int k_tap = 0; k_tap < maxk; k_tap++)
                    {
                        const int kd = k_tap / (kernel_h * kernel_w);
                        const int ky = (k_tap / kernel_w) % kernel_h;
                        const int kx = k_tap % kernel_w;

                        for (int c = 0; c < num_input; c++)
                        {
                            const float* sptr = bottom_blob_bordered.channel(c / 4);
                            const int cl = c % 4;
                            float* dptr = aptr + (size_t)(k_tap * num_input + c) * 4;

                            for (int t = 0; t < 4; t++)
                            {
                                const int s = q * 4 + t;
                                const int x = s % outw;
                                const int y = (s / outw) % outh;
                                const int z = s / (outw * outh);

                                const int ix = x + kx;
                                const int iy = y + ky;
                                const int iz = z + kd;

                                dptr[t] = sptr[((((size_t)iz * bh + iy) * bw + ix) * 4) + cl];
                            }
                        }
                    }
                }

                // weight vec4 for tap k_tap, input pack q, output pack p, input
                // lane l holds output channels 4p..4p+3 at bt index
                // k_tap*num_input+q*4+l; tm strides (q,p) by outpack*16 so copy
                // element by element. tm already has the flipped taps
                for (int p = 0; p < outpack; p++)
                {
                    float* btptr = BT.row(p);

                    for (int k_tap = 0; k_tap < maxk; k_tap++)
                    {
                        for (int q = 0; q < inpack; q++)
                        {
                            for (int l = 0; l < 4; l++)
                            {
                                const float* kptr = tm + (size_t)k_tap * blocksize + ((size_t)q * outpack + p) * 16 + l * 4;
                                float* dptr = btptr + (size_t)(k_tap * num_input + q * 4 + l) * 4;

                                for (int t = 0; t < 4; t++)
                                    dptr[t] = kptr[t];
                            }
                        }
                    }
                }

                if (gemm_transB_pack4_spu(A, BT, C, 1.f, 0.f, 0.f, 0) == 0)
                {
                    // C row q, float j*4+t is output voxel 4q+t of output channel
                    // j; the top blob stores voxel s as the vec4 at channel(p)
                    // voxel s with lane t2 = output channel 4p+t2
                    for (int p = 0; p < outpack; p++)
                    {
                        float* outptr = top_blob.channel(p);

                        v4f32 _bias = vec_splats(0.f);
                        if (bias_term)
                            _bias = __ncnn_altivec_load((const float*)bias_data + p * 4);

                        for (int s = 0; s < S; s++)
                        {
                            const int q = s / 4;
                            const int t = s % 4;
                            const float* cptr = C.row(q);

                            float _v[4];
                            for (int t2 = 0; t2 < 4; t2++)
                                _v[t2] = cptr[(size_t)(4 * p + t2) * 4 + t];

                            v4f32 _sum = vec_add(__ncnn_altivec_load(_v), _bias);
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

                float* outptr = top_blob.channel(p);

                for (int z = 0; z < outd; z++)
                {
                    for (int y = 0; y < outh; y++)
                    {
                        for (int x = 0; x < outw; x++)
                        {
                            v4f32 _sum = _bias;

                            for (int q = 0; q < inpack; q++)
                            {
                                const float* inptr = bottom_blob_bordered.channel(q);

                                for (int kd = 0; kd < kernel_d; kd++)
                                {
                                    for (int ky = 0; ky < kernel_h; ky++)
                                    {
                                        const float* sptr = inptr + (((size_t)(z + kd) * bh + (y + ky)) * bw + x) * 4;

                                        for (int kx = 0; kx < kernel_w; kx++)
                                        {
                                            const int k = (kd * kernel_h + ky) * kernel_w + kx;

                                            v4f32 _val = __ncnn_altivec_load(sptr + kx * 4);
                                            const float* kptr = tm + (size_t)k * blocksize + ((size_t)q * outpack + p) * 16;

                                            _sum = vec_madd(vec_splat(_val, 0), __ncnn_altivec_load(kptr + 0), _sum);
                                            _sum = vec_madd(vec_splat(_val, 1), __ncnn_altivec_load(kptr + 4), _sum);
                                            _sum = vec_madd(vec_splat(_val, 2), __ncnn_altivec_load(kptr + 8), _sum);
                                            _sum = vec_madd(vec_splat(_val, 3), __ncnn_altivec_load(kptr + 12), _sum);
                                        }
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

        return Deconvolution3D::forward(bottom_blob_unpacked, top_blob, opt);
    }

    return forward_generic(bottom_blob, top_blob, opt);
}

} // namespace ncnn
