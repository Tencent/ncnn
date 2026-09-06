// Copyright 2026 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause
#include "inversespectrogram_x86.h"

#include "gemm.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"
#include "cpu.h"

namespace ncnn {

InverseSpectrogram_x86::InverseSpectrogram_x86()
{
    gemm_real = 0;
    gemm_imag = 0;
}

int InverseSpectrogram_x86::create_pipeline(const Option& opt)
{
    // build inverse dft basis
    // real: re[i] = sum_k (sp_re[k] * cos + sp_im[k] * (-sin)) / n_fft
    // imag: im[i] = sum_k (sp_re[k] * sin + sp_im[k] * ( cos)) / n_fft
    //
    // merge sp_re and sp_im as B = [sp_re, sp_im] length 2*n_fft
    // A_real[i, k]       = cos(2pi*i*k/n_fft) / n_fft * window[i]
    // A_real[i, k+n_fft] = -sin(2pi*i*k/n_fft) / n_fft * window[i]
    //
    // A_imag[i, k]       = sin(2pi*i*k/n_fft) / n_fft * window[i]
    // A_imag[i, k+n_fft] = cos(2pi*i*k/n_fft) / n_fft * window[i]

    const int K = 2 * n_fft;

    basis_data_real.create(K, n_fft, (size_t)4u, 1);
    basis_data_imag.create(K, n_fft, (size_t)4u, 1);
    if (basis_data_real.empty() || basis_data_imag.empty())
        return -100;

    for (int i = 0; i < n_fft; i++)
    {
        float* real_row = basis_data_real.row<float>(i);
        float* imag_row = basis_data_imag.row<float>(i);

        float scale = window_data[i] / n_fft;

        for (int k = 0; k < n_fft; k++)
        {
            double angle = 2 * 3.14159265358979323846 * i * k / n_fft;

            float c = (float)cos(angle) * scale;
            float s = (float)sin(angle) * scale;

            // [sp_re, sp_im]
            real_row[k] = c;
            real_row[k + n_fft] = -s;

            imag_row[k] = s;
            imag_row[k + n_fft] = c;
        }
    }

    gemm_real = create_layer_cpu("Gemm");
    {
        ncnn::ParamDict gemm_pd;

        gemm_pd.set(0, 1.f);   // alpha
        gemm_pd.set(1, 1.f);   // beta
        gemm_pd.set(2, 0);     // transA
        gemm_pd.set(3, 1);     // transB (B is BT layout already)
        gemm_pd.set(4, 1);     // constantA
        gemm_pd.set(5, 0);     // constantB
        gemm_pd.set(6, 0);     // constantC
        gemm_pd.set(7, n_fft); // constantM
        gemm_pd.set(8, 0);     // constantN
        gemm_pd.set(9, K);     // constantK
        gemm_pd.set(14, 0);    // output_transpose

        gemm_real->load_param(gemm_pd);

        // set constant A directly
        ((Gemm*)gemm_real)->A_data = basis_data_real;

        Option opt_g = opt;
        opt_g.use_packing_layout = false;

        gemm_real->create_pipeline(opt_g);
    }

    gemm_imag = create_layer_cpu("Gemm");
    {
        ncnn::ParamDict gemm_pd;

        gemm_pd.set(0, 1.f);   // alpha
        gemm_pd.set(1, 1.f);   // beta
        gemm_pd.set(2, 0);     // transA
        gemm_pd.set(3, 1);     // transB (B is BT layout already)
        gemm_pd.set(4, 1);     // constantA
        gemm_pd.set(5, 0);     // constantB
        gemm_pd.set(6, 0);     // constantC
        gemm_pd.set(7, n_fft); // constantM
        gemm_pd.set(8, 0);     // constantN
        gemm_pd.set(9, K);     // constantK
        gemm_pd.set(14, 0);    // output_transpose

        gemm_imag->load_param(gemm_pd);

        // set constant A directly
        ((Gemm*)gemm_imag)->A_data = basis_data_imag;

        Option opt_g = opt;
        opt_g.use_packing_layout = false;

        gemm_imag->create_pipeline(opt_g);
    }

    return 0;
}

int InverseSpectrogram_x86::destroy_pipeline(const Option& opt)
{
    if (gemm_real)
    {
        gemm_real->destroy_pipeline(opt);
        delete gemm_real;
        gemm_real = 0;
    }

    if (gemm_imag)
    {
        gemm_imag->destroy_pipeline(opt);
        delete gemm_imag;
        gemm_imag = 0;
    }

    return 0;
}

int InverseSpectrogram_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py#L630

    // TODO custom window
    // TODO output length

    const int frames = bottom_blob.h;
    const int freqs = bottom_blob.c;
    // assert freqs == n_fft or freqs == n_fft / 2 + 1

    const int onesided = freqs == n_fft / 2 + 1 ? 1 : 0;

    const int outsize = center ? (frames - 1) * hoplen + (n_fft - n_fft / 2 * 2) : (frames - 1) * hoplen + n_fft;

    const size_t elemsize = bottom_blob.elemsize;

    if (elemsize != sizeof(float))
        return -100;

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

    // build B (frames, 2*n_fft)
    Mat B(2 * n_fft, frames, elemsize, opt.workspace_allocator);
    if (B.empty())
        return -100;

    for (int j = 0; j < frames; j++)
    {
        float* bptr = B.row<float>(j);

        if (onesided == 1)
        {
            for (int k = 0; k < n_fft / 2 + 1; k++)
            {
                bptr[k] = bottom_blob.channel(k).row(j)[0];
                bptr[k + n_fft] = bottom_blob.channel(k).row(j)[1];
            }
            for (int k = n_fft / 2 + 1; k < n_fft; k++)
            {
                bptr[k] = bottom_blob.channel(n_fft - k).row(j)[0];
                bptr[k + n_fft] = -bottom_blob.channel(n_fft - k).row(j)[1];
            }
        }
        else
        {
            for (int k = 0; k < n_fft; k++)
            {
                bptr[k] = bottom_blob.channel(k).row(j)[0];
                bptr[k + n_fft] = bottom_blob.channel(k).row(j)[1];
            }
        }
    }

    // gemm to get time-domain frames (n_fft, frames)
    // output shape: w=frames h=n_fft
    Mat Yre;
    Mat Yim;
    {
        std::vector<Mat> inputs;
        inputs.push_back(B);

        std::vector<Mat> outputs;
        outputs.push_back(Mat());

        Option opt_g = opt;
        opt_g.use_packing_layout = false;
        opt_g.blob_allocator = opt.workspace_allocator;

        int ret = gemm_real->forward(inputs, outputs, opt_g);
        if (ret != 0)
            return ret;

        Yre = outputs[0];
    }
    {
        std::vector<Mat> inputs;
        inputs.push_back(B);

        std::vector<Mat> outputs;
        outputs.push_back(Mat());

        Option opt_g = opt;
        opt_g.use_packing_layout = false;
        opt_g.blob_allocator = opt.workspace_allocator;

        int ret = gemm_imag->forward(inputs, outputs, opt_g);
        if (ret != 0)
            return ret;

        Yim = outputs[0];
    }

    // overlap-add
    for (int j = 0; j < frames; j++)
    {
        for (int i = 0; i < n_fft; i++)
        {
            // Yre/Yim layout: row(i) length=frames, element [j] is frame j
            float re = Yre.row<const float>(i)[j];
            float im = Yim.row<const float>(i)[j];

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
        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < outsize; i += 16)
        {
            __m512 wss = _mm512_loadu_ps((const float*)window_sumsquare + i);
            __mmask16 mask = _mm512_cmp_ps_mask(wss, _mm512_setzero_ps(), _MM_CMPINT_NE);
            if (mask)
            {
                __m512 inv_wss = _mm512_div_ps(_mm512_set1_ps(1.0f), wss);

                float re_buf[16], im_buf[16];
                for (int k = 0; k < 16; k++)
                {
                    re_buf[k] = top_blob.row(i + k)[0];
                    im_buf[k] = top_blob.row(i + k)[1];
                }

                __m512 re_vals = _mm512_loadu_ps(re_buf);
                __m512 im_vals = _mm512_loadu_ps(im_buf);

                re_vals = _mm512_mul_ps(re_vals, inv_wss);
                im_vals = _mm512_mul_ps(im_vals, inv_wss);

                _mm512_storeu_ps(re_buf, re_vals);
                _mm512_storeu_ps(im_buf, im_vals);

                for (int k = 0; k < 16; k++)
                {
                    if (((const float*)window_sumsquare)[i + k] != 0.f)
                    {
                        top_blob.row(i + k)[0] = re_buf[k];
                        top_blob.row(i + k)[1] = im_buf[k];
                    }
                }
            }
        }
#endif // __AVX512F__
        for (; i + 7 < outsize; i += 8)
        {
            __m256 wss = _mm256_loadu_ps((const float*)window_sumsquare + i);
            __m256 mask = _mm256_cmp_ps(wss, _mm256_setzero_ps(), _MM_CMPINT_NE);
            if (_mm256_movemask_ps(mask))
            {
                __m256 inv_wss = _mm256_div_ps(_mm256_set1_ps(1.0f), wss);

                float re_buf[8], im_buf[8];
                for (int k = 0; k < 8; k++)
                {
                    re_buf[k] = top_blob.row(i + k)[0];
                    im_buf[k] = top_blob.row(i + k)[1];
                }

                __m256 re_vals = _mm256_loadu_ps(re_buf);
                __m256 im_vals = _mm256_loadu_ps(im_buf);

                re_vals = _mm256_mul_ps(re_vals, inv_wss);
                im_vals = _mm256_mul_ps(im_vals, inv_wss);

                _mm256_storeu_ps(re_buf, re_vals);
                _mm256_storeu_ps(im_buf, im_vals);

                for (int k = 0; k < 8; k++)
                {
                    if (((const float*)window_sumsquare)[i + k] != 0.f)
                    {
                        top_blob.row(i + k)[0] = re_buf[k];
                        top_blob.row(i + k)[1] = im_buf[k];
                    }
                }
            }
        }
#endif // __AVX__
        for (; i + 3 < outsize; i += 4)
        {
            __m128 wss = _mm_loadu_ps((const float*)window_sumsquare + i);
            __m128 mask = _mm_cmpneq_ps(wss, _mm_setzero_ps());
            if (_mm_movemask_ps(mask))
            {
                __m128 inv_wss = _mm_div_ps(_mm_set1_ps(1.0f), wss);

                float re_buf[4], im_buf[4];
                for (int k = 0; k < 4; k++)
                {
                    re_buf[k] = top_blob.row(i + k)[0];
                    im_buf[k] = top_blob.row(i + k)[1];
                }

                __m128 re_vals = _mm_loadu_ps(re_buf);
                __m128 im_vals = _mm_loadu_ps(im_buf);

                re_vals = _mm_mul_ps(re_vals, inv_wss);
                im_vals = _mm_mul_ps(im_vals, inv_wss);

                _mm_storeu_ps(re_buf, re_vals);
                _mm_storeu_ps(im_buf, im_vals);

                for (int k = 0; k < 4; k++)
                {
                    if (((const float*)window_sumsquare)[i + k] != 0.f)
                    {
                        top_blob.row(i + k)[0] = re_buf[k];
                        top_blob.row(i + k)[1] = im_buf[k];
                    }
                }
            }
        }
#endif // __SSE2__
        for (; i < outsize; i++)
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
        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < outsize; i += 16)
        {
            __m512 wss = _mm512_loadu_ps((const float*)window_sumsquare + i);
            __mmask16 mask = _mm512_cmp_ps_mask(wss, _mm512_setzero_ps(), _MM_CMPINT_NE);
            if (mask)
            {
                __m512 inv_wss = _mm512_div_ps(_mm512_set1_ps(1.0f), wss);
                __m512 out_vals = _mm512_loadu_ps((const float*)top_blob + i);
                out_vals = _mm512_mul_ps(out_vals, inv_wss);
                _mm512_storeu_ps((float*)top_blob + i, out_vals);
            }
        }
#endif // __AVX512F__
        for (; i + 7 < outsize; i += 8)
        {
            __m256 wss = _mm256_loadu_ps((const float*)window_sumsquare + i);
            __m256 mask = _mm256_cmp_ps(wss, _mm256_setzero_ps(), _MM_CMPINT_NE);
            if (_mm256_movemask_ps(mask))
            {
                __m256 inv_wss = _mm256_div_ps(_mm256_set1_ps(1.0f), wss);
                __m256 out_vals = _mm256_loadu_ps((const float*)top_blob + i);
                out_vals = _mm256_mul_ps(out_vals, inv_wss);
                _mm256_storeu_ps((float*)top_blob + i, out_vals);
            }
        }
#endif // __AVX__
        for (; i + 3 < outsize; i += 4)
        {
            __m128 wss = _mm_loadu_ps((const float*)window_sumsquare + i);
            __m128 mask = _mm_cmpneq_ps(wss, _mm_setzero_ps());
            if (_mm_movemask_ps(mask))
            {
                __m128 inv_wss = _mm_div_ps(_mm_set1_ps(1.0f), wss);
                __m128 out_vals = _mm_loadu_ps((const float*)top_blob + i);
                out_vals = _mm_mul_ps(out_vals, inv_wss);
                _mm_storeu_ps((float*)top_blob + i, out_vals);
            }
        }
#endif // __SSE2__
        for (; i < outsize; i++)
        {
            if (window_sumsquare[i] != 0.f)
                top_blob[i] /= window_sumsquare[i];
        }
    }

    return 0;
}

} // namespace ncnn