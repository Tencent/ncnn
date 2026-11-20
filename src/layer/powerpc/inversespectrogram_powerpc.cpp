// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "inversespectrogram_powerpc.h"

#include <limits.h>

#if __ALTIVEC__
#include "powerpc_mathfun.h"
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

InverseSpectrogram_powerpc::InverseSpectrogram_powerpc()
{
    // base defaults kept: one_blob_only = true, support_inplace = false
    // the base class is not packing aware, support_packing stays false
}

#if __ALTIVEC__

// {r0,i0,r1,i1} {r2,i2,r3,i3} -> {r0,r1,r2,r3} {i0,i1,i2,i3}
static NCNN_FORCEINLINE void deinterleave_ri(v4f32 _x0, v4f32 _x1, v4f32& _r, v4f32& _i)
{
    // float lanes are 4 bytes, gather the even and odd float lanes across two vectors
    const v16u8 _perm_r = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
    const v16u8 _perm_i = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};

    _r = vec_perm(_x0, _x1, _perm_r);
    _i = vec_perm(_x0, _x1, _perm_i);
}

#endif // __ALTIVEC__

int InverseSpectrogram_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elembits() != 32)
    {
        // int8 / bf16, fall back to the generic implementation
        return InverseSpectrogram::forward(bottom_blob, top_blob, opt);
    }

#if __ALTIVEC__
    if (bottom_blob.elempack == 1)
    {
        const int frames = bottom_blob.h;
        const int freqs = bottom_blob.c;
        // assert freqs == n_fft or freqs == n_fft / 2 + 1

        const int onesided = freqs == n_fft / 2 + 1 ? 1 : 0;

        const int outsize = center ? (frames - 1) * hoplen + (n_fft - n_fft / 2 * 2) : (frames - 1) * hoplen + n_fft;

        const size_t elemsize = bottom_blob.elemsize;

        if (returns == 0)
        {
            top_blob.create(2, outsize, elemsize, opt.blob_allocator);
        }
        else
        {
            top_blob.create(outsize, elemsize, opt.blob_allocator);
        }
        if (top_blob.empty())
            return -100;

        Mat window_sumsquare(outsize + n_fft, elemsize, opt.workspace_allocator);
        if (window_sumsquare.empty())
            return -100;

        top_blob.fill(0.f);
        window_sumsquare.fill(0.f);

        const float norm1 = normalized == 1 ? sqrtf(n_fft) : 1.f;
        const float norm2 = normalized == 2 ? window_data[n_fft] : 1.f;
        const float norm = normalized != 0 ? (normalized == 1 ? norm1 : norm2) : 1.f;

        for (int j = 0; j < frames; j++)
        {
            // collect complex
            Mat sp(2, n_fft);
            if (onesided == 1)
            {
                for (int k = 0; k < n_fft / 2 + 1; k++)
                {
                    sp.row(k)[0] = bottom_blob.channel(k).row(j)[0];
                    sp.row(k)[1] = bottom_blob.channel(k).row(j)[1];
                }
                for (int k = n_fft / 2 + 1; k < n_fft; k++)
                {
                    sp.row(k)[0] = bottom_blob.channel(n_fft - k).row(j)[0];
                    sp.row(k)[1] = -bottom_blob.channel(n_fft - k).row(j)[1];
                }
            }
            else
            {
                for (int k = 0; k < n_fft; k++)
                {
                    sp.row(k)[0] = bottom_blob.channel(k).row(j)[0];
                    sp.row(k)[1] = bottom_blob.channel(k).row(j)[1];
                }
            }

            if (normalized == 1 || normalized == 2)
            {
                // 2 * n_fft is always even
                float* sp_ptr = sp;
                int i = 0;
                for (; i + 3 < 2 * n_fft; i += 4)
                {
                    __ncnn_altivec_store(sp_ptr + i, __ncnn_altivec_mul(__ncnn_altivec_load(sp_ptr + i), vec_splats(norm)));
                }
                for (; i < 2 * n_fft; i++)
                {
                    sp_ptr[i] *= norm;
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < n_fft; i++)
            {
                // inverse dft
                v4f32 _re = vec_splats(0.f);
                v4f32 _im = vec_splats(0.f);

                const v4f32 _freq_step = vec_splats(2 * 3.14159265358979323846f * i / n_fft);
                const float* sp_ptr = sp;

                int k = 0;
                for (; k + 3 < n_fft; k += 4)
                {
                    v4f32 _x0 = __ncnn_altivec_load(sp_ptr + k * 2);
                    v4f32 _x1 = __ncnn_altivec_load(sp_ptr + k * 2 + 4);

                    v4f32 _r;
                    v4f32 _imsp;
                    deinterleave_ri(_x0, _x1, _r, _imsp);

                    const v4i32 _k = (v4i32){k, k + 1, k + 2, k + 3};
                    v4f32 _angle = __ncnn_altivec_mul(__ncnn_altivec_ffint_s(_k), _freq_step);

                    v4f32 _c;
                    v4f32 _s;
                    sincos_ps(_angle, _s, _c);

                    // re += r * cos - i * sin
                    _re = vec_madd(_r, _c, vec_nmsub(_imsp, _s, _re));
                    // im += r * sin + i * cos
                    _im = vec_madd(_imsp, _c, vec_madd(_r, _s, _im));
                }

                float re = __ncnn_altivec_reduce_add(_re);
                float im = __ncnn_altivec_reduce_add(_im);

                for (; k < n_fft; k++)
                {
                    double angle = 2 * 3.14159265358979323846 * i * k / n_fft;

                    re += sp.row(k)[0] * cosf(angle) - sp.row(k)[1] * sinf(angle);
                    im += sp.row(k)[0] * sinf(angle) + sp.row(k)[1] * cosf(angle);
                }

                re /= n_fft;
                im /= n_fft;

                // apply window
                re *= window_data[i];
                im *= window_data[i];

                int output_index = j * hoplen + i;
                if (center == 1)
                {
                    output_index -= n_fft / 2;
                }
                if (output_index >= 0 && output_index < outsize)
                {
                    // square window
                    window_sumsquare[output_index] += window_data[i] * window_data[i];

                    if (returns == 0)
                    {
                        top_blob.row(output_index)[0] += re;
                        top_blob.row(output_index)[1] += im;
                    }
                    if (returns == 1)
                    {
                        top_blob[output_index] += re;
                    }
                    if (returns == 2)
                    {
                        top_blob[output_index] += im;
                    }
                }
            }
        }

        // square window norm
        if (returns == 0)
        {
            for (int i = 0; i < outsize; i++)
            {
                if (window_sumsquare[i] != 0.f)
                {
                    top_blob.row(i)[0] /= window_sumsquare[i];
                    top_blob.row(i)[1] /= window_sumsquare[i];
                }
            }
        }
        else
        {
            for (int i = 0; i < outsize; i++)
            {
                if (window_sumsquare[i] != 0.f)
                    top_blob[i] /= window_sumsquare[i];
            }
        }

        return 0;
    }

    // packed fp32 input, unpack and use the generic implementation
    if (bottom_blob.elempack != 1)
    {
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        if (bottom_blob_unpacked.empty())
            return -100;

        return InverseSpectrogram::forward(bottom_blob_unpacked, top_blob, opt);
    }
#endif // __ALTIVEC__

    return InverseSpectrogram::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
