// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "spectrogram_powerpc.h"

#include <limits.h>

#if __ALTIVEC__
#include "powerpc_mathfun.h"
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Spectrogram_powerpc::Spectrogram_powerpc()
{
    // base defaults kept: one_blob_only = true, support_inplace = false
    // the base class is not packing aware, support_packing stays false
}

#if __ALTIVEC__

static void spectrogram_forward_altivec(const Mat& bottom_blob_bordered, Mat& top_blob, const Mat& window_data,
                                        int n_fft, int power, int hoplen, int normalized, int num_threads)
{
    const int size = bottom_blob_bordered.w;

    const int frames = (size - n_fft) / hoplen + 1;
    const int freqs_onesided = n_fft / 2 + 1;

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < freqs_onesided; i++)
    {
        const float* ptr = bottom_blob_bordered;
        float* outptr = power == 0 ? top_blob.channel(i) : top_blob.row(i);

        const v4f32 _freq_step = vec_splats(2 * 3.14159265358979323846f * i / n_fft);

        for (int j = 0; j < frames; j++)
        {
            v4f32 _re = vec_splats(0.f);
            v4f32 _im = vec_splats(0.f);

            int k = 0;
            for (; k + 3 < n_fft; k += 4)
            {
                // apply window
                v4f32 _v = __ncnn_altivec_load(ptr + k);
                _v = __ncnn_altivec_mul(_v, __ncnn_altivec_load(window_data.row(0) + k));

                // dft
                const v4i32 _k = (v4i32) {
                    k, k + 1, k + 2, k + 3
                };
                v4f32 _angle = __ncnn_altivec_mul(__ncnn_altivec_ffint_s(_k), _freq_step);

                v4f32 _c;
                v4f32 _s;
                sincos_ps(_angle, _s, _c);

                _re = vec_madd(_v, _c, _re);
                _im = vec_nmsub(_v, _s, _im);
            }

            float re = __ncnn_altivec_reduce_add(_re);
            float im = __ncnn_altivec_reduce_add(_im);

            for (; k < n_fft; k++)
            {
                float v = ptr[k];

                // apply window
                v *= window_data[k];

                // dft
                double angle = 2 * 3.14159265358979323846 * i * k / n_fft;

                re += v * cosf(angle); // + imag * sinf(angle);
                im -= v * sinf(angle); // + imag * cosf(angle);
            }

            if (normalized == 1)
            {
                float norm = 1.f / sqrtf(n_fft);
                re *= norm;
                im *= norm;
            }
            if (normalized == 2)
            {
                float norm = window_data[n_fft];
                re *= norm;
                im *= norm;
            }

            if (power == 0)
            {
                // complex as real
                outptr[0] = re;
                outptr[1] = im;
                outptr += 2;
            }
            if (power == 1)
            {
                // magnitude
                outptr[0] = sqrtf(re * re + im * im);
                outptr += 1;
            }
            if (power == 2)
            {
                outptr[0] = re * re + im * im;
                outptr += 1;
            }

            ptr += hoplen;
        }
    }
}

#endif // __ALTIVEC__

int Spectrogram_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elembits() != 32)
    {
        // int8 / bf16, fall back to the generic implementation
        return Spectrogram::forward(bottom_blob, top_blob, opt);
    }

#if __ALTIVEC__
    if (bottom_blob.elempack == 1)
    {
        // same center padding as the generic implementation
        Mat bottom_blob_bordered = bottom_blob;
        if (center == 1)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            if (pad_type == 0)
                copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, n_fft / 2, n_fft / 2, BORDER_CONSTANT, 0.f, opt_b);
            if (pad_type == 1)
                copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, n_fft / 2, n_fft / 2, BORDER_REPLICATE, 0.f, opt_b);
            if (pad_type == 2)
                copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, n_fft / 2, n_fft / 2, BORDER_REFLECT, 0.f, opt_b);
        }

        const int size = bottom_blob_bordered.w;

        const int frames = (size - n_fft) / hoplen + 1;
        const int freqs_onesided = n_fft / 2 + 1;
        const int freqs = onesided ? freqs_onesided : n_fft;

        const size_t elemsize = bottom_blob_bordered.elemsize;

        if (power == 0)
        {
            top_blob.create(2, frames, freqs, elemsize, opt.blob_allocator);
        }
        else
        {
            top_blob.create(frames, freqs, elemsize, opt.blob_allocator);
        }
        if (top_blob.empty())
            return -100;

        spectrogram_forward_altivec(bottom_blob_bordered, top_blob, window_data,
                                    n_fft, power, hoplen, normalized, opt.num_threads);

        if (!onesided)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = freqs_onesided; i < n_fft; i++)
            {
                if (power == 0)
                {
                    const float* ptr = top_blob.channel(n_fft - i);
                    float* outptr = top_blob.channel(i);

                    for (int j = 0; j < frames; j++)
                    {
                        // complex as real
                        outptr[0] = ptr[0];
                        outptr[1] = -ptr[1];
                        ptr += 2;
                        outptr += 2;
                    }
                }
                else // if (power == 1 || power == 2)
                {
                    const float* ptr = top_blob.row(n_fft - i);
                    float* outptr = top_blob.row(i);

                    memcpy(outptr, ptr, frames * sizeof(float));
                }
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

        return Spectrogram::forward(bottom_blob_unpacked, top_blob, opt);
    }
#endif // __ALTIVEC__

    return Spectrogram::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
