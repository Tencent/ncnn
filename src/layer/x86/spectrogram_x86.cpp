// Copyright 2026 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause
#include "spectrogram_x86.h"

#include "unfold.h"
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

Spectrogram_x86::Spectrogram_x86()
{
    unfold = 0;
    gemm = 0;
}

int Spectrogram_x86::create_pipeline(const Option& opt)
{
    if (onesided)
    {
        n_freq = n_fft / 2 + 1;
    }
    else
    {
        n_freq = n_fft;
    }

    // basis_data (2 * n_freq, n_fft)
    // first  n_freq rows : cos
    // second n_freq rows : -sin
    basis_data.create(n_fft, 2 * n_freq, (size_t)4u, 1);
    if (basis_data.empty())
        return -100;

    for (int i = 0; i < n_freq; i++)
    {
        float* real_row = basis_data.row<float>(i);
        float* imag_row = basis_data.row<float>(i + n_freq);

        for (int j = 0; j < n_fft; j++)
        {
            double angle = 2 * 3.14159265358979323846 * i * j / n_fft;

            // multiply window
            float w = window_data[j];

            real_row[j] = (float)cos(angle) * w;
            imag_row[j] = (float)(-sin(angle)) * w;
        }
    }

    unfold = create_layer_cpu("Unfold");
    {
        ncnn::ParamDict unfold_pd;

        unfold_pd.set(1, n_fft);  // kernel_w
        unfold_pd.set(11, 1);     // kernel_h
        unfold_pd.set(2, 1);      // dilation_w
        unfold_pd.set(12, 1);     // dilation_h
        unfold_pd.set(3, hoplen); // stride_w
        unfold_pd.set(13, 1);     // stride_h
        unfold_pd.set(4, 0);      // pad_left
        unfold_pd.set(15, 0);     // pad_right
        unfold_pd.set(14, 0);     // pad_top
        unfold_pd.set(16, 0);     // pad_bottom
        unfold_pd.set(18, 0.f);   // pad_value

        unfold->load_param(unfold_pd);

        Option opt_unfold = opt;
        opt_unfold.use_packing_layout = false;

        unfold->create_pipeline(opt_unfold);
    }

    gemm = create_layer_cpu("Gemm");
    {
        ncnn::ParamDict gemm_pd;

        gemm_pd.set(0, 1.f);        // alpha
        gemm_pd.set(1, 1.f);        // beta
        gemm_pd.set(2, 0);          // transA
        gemm_pd.set(3, 0);          // transB
        gemm_pd.set(4, 1);          // constantA
        gemm_pd.set(5, 0);          // constantB
        gemm_pd.set(6, 0);          // constantC
        gemm_pd.set(7, 2 * n_freq); // constantM
        gemm_pd.set(8, 0);          // constantN
        gemm_pd.set(9, n_fft);      // constantK
        gemm_pd.set(14, 0);         // output_transpose

        gemm->load_param(gemm_pd);

        // set constant A directly
        ((Gemm*)gemm)->A_data = basis_data;

        Option opt_gemm = opt;
        opt_gemm.use_packing_layout = false;

        gemm->create_pipeline(opt_gemm);
    }

    return 0;
}

int Spectrogram_x86::destroy_pipeline(const Option& opt)
{
    if (unfold)
    {
        unfold->destroy_pipeline(opt);
        delete unfold;
        unfold = 0;
    }

    if (gemm)
    {
        gemm->destroy_pipeline(opt);
        delete gemm;
        gemm = 0;
    }

    return 0;
}

int Spectrogram_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
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

    const size_t elemsize = bottom_blob_bordered.elemsize;

    if (elemsize != sizeof(float))
    {
        return -100;
    }

    if (power == 0)
    {
        top_blob.create(2, frames, n_freq, elemsize, opt.blob_allocator);
    }
    else
    {
        top_blob.create(frames, n_freq, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    // unfold to columns (frames, n_fft)
    Mat cols;
    {
        Option opt_u = opt;
        opt_u.blob_allocator = opt.workspace_allocator;
        opt_u.use_packing_layout = false;

        int ret = unfold->forward(bottom_blob_bordered, cols, opt_u);
        if (ret != 0)
            return ret;
    }

    // gemm: (2*n_freq, n_fft) x (n_fft, frames) = (2*n_freq, frames)
    Mat Y;
    {
        std::vector<Mat> inputs;
        inputs.push_back(cols);

        std::vector<Mat> outputs;
        outputs.push_back(Mat());

        Option opt_g = opt;
        opt_g.use_packing_layout = false;

        int ret = gemm->forward(inputs, outputs, opt_g);
        if (ret != 0)
            return ret;

        Y = outputs[0];
    }

    const float* y = Y;

    if (power == 0) // as complex
    {
        // copy
        for (int i = 0; i < frames; i++)
        {
            for (int j = 0; j < n_freq; j++)
            {
                top_blob.channel(j).row<float>(i)[0] = y[j * frames + i];
                top_blob.channel(j).row<float>(i)[1] = y[(j + n_freq) * frames + i];
            }
        }
    }
    else
    {
        if (power == 1) // magnitude sqrt(re * re + im * im);
        {
            // copy with simd optimization
            int total = frames * n_freq;
            int i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < total; i += 16)
            {
                __m512 re_vals = _mm512_setzero_ps();
                __m512 im_vals = _mm512_setzero_ps();
                
                // gather real and imaginary parts
                // process 16 elements at a time
                float re_buf[16], im_buf[16];
                for (int k = 0; k < 16; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    re_buf[k] = y[idx_j * frames + idx_i];
                    im_buf[k] = y[(idx_j + n_freq) * frames + idx_i];
                }
                re_vals = _mm512_loadu_ps(re_buf);
                im_vals = _mm512_loadu_ps(im_buf);
                
                __m512 sq = _mm512_add_ps(_mm512_mul_ps(re_vals, re_vals), _mm512_mul_ps(im_vals, im_vals));
                __m512 mag = _mm512_sqrt_ps(sq);
                
                float out_buf[16];
                _mm512_storeu_ps(out_buf, mag);
                
                for (int k = 0; k < 16; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    top_blob.row<float>(idx_j)[idx_i] = out_buf[k];
                }
            }
#endif // __AVX512F__
            for (; i + 7 < total; i += 8)
            {
                __m256 re_vals = _mm256_setzero_ps();
                __m256 im_vals = _mm256_setzero_ps();
                
                float re_buf[8], im_buf[8];
                for (int k = 0; k < 8; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    re_buf[k] = y[idx_j * frames + idx_i];
                    im_buf[k] = y[(idx_j + n_freq) * frames + idx_i];
                }
                re_vals = _mm256_loadu_ps(re_buf);
                im_vals = _mm256_loadu_ps(im_buf);
                
                __m256 sq = _mm256_add_ps(_mm256_mul_ps(re_vals, re_vals), _mm256_mul_ps(im_vals, im_vals));
                __m256 mag = _mm256_sqrt_ps(sq);
                
                float out_buf[8];
                _mm256_storeu_ps(out_buf, mag);
                
                for (int k = 0; k < 8; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    top_blob.row<float>(idx_j)[idx_i] = out_buf[k];
                }
            }
#endif // __AVX__
            for (; i + 3 < total; i += 4)
            {
                __m128 re_vals = _mm_setzero_ps();
                __m128 im_vals = _mm_setzero_ps();
                
                float re_buf[4], im_buf[4];
                for (int k = 0; k < 4; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    re_buf[k] = y[idx_j * frames + idx_i];
                    im_buf[k] = y[(idx_j + n_freq) * frames + idx_i];
                }
                re_vals = _mm_loadu_ps(re_buf);
                im_vals = _mm_loadu_ps(im_buf);
                
                __m128 sq = _mm_add_ps(_mm_mul_ps(re_vals, re_vals), _mm_mul_ps(im_vals, im_vals));
                __m128 mag = _mm_sqrt_ps(sq);
                
                float out_buf[4];
                _mm_storeu_ps(out_buf, mag);
                
                for (int k = 0; k < 4; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    top_blob.row<float>(idx_j)[idx_i] = out_buf[k];
                }
            }
#endif // __SSE2__
            for (; i < total; i++)
            {
                int idx_i = i % frames;
                int idx_j = i / frames;
                float re = y[idx_j * frames + idx_i];
                float im = y[(idx_j + n_freq) * frames + idx_i];
                top_blob.row<float>(idx_j)[idx_i] = sqrtf(re * re + im * im);
            }
        }
        else if (power == 2) // power re * re + im * im;
        {
            // copy with simd optimization
            int total = frames * n_freq;
            int i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < total; i += 16)
            {
                __m512 re_vals = _mm512_setzero_ps();
                __m512 im_vals = _mm512_setzero_ps();
                
                float re_buf[16], im_buf[16];
                for (int k = 0; k < 16; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    re_buf[k] = y[idx_j * frames + idx_i];
                    im_buf[k] = y[(idx_j + n_freq) * frames + idx_i];
                }
                re_vals = _mm512_loadu_ps(re_buf);
                im_vals = _mm512_loadu_ps(im_buf);
                
                __m512 sq = _mm512_add_ps(_mm512_mul_ps(re_vals, re_vals), _mm512_mul_ps(im_vals, im_vals));
                
                float out_buf[16];
                _mm512_storeu_ps(out_buf, sq);
                
                for (int k = 0; k < 16; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    top_blob.row<float>(idx_j)[idx_i] = out_buf[k];
                }
            }
#endif // __AVX512F__
            for (; i + 7 < total; i += 8)
            {
                __m256 re_vals = _mm256_setzero_ps();
                __m256 im_vals = _mm256_setzero_ps();
                
                float re_buf[8], im_buf[8];
                for (int k = 0; k < 8; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    re_buf[k] = y[idx_j * frames + idx_i];
                    im_buf[k] = y[(idx_j + n_freq) * frames + idx_i];
                }
                re_vals = _mm256_loadu_ps(re_buf);
                im_vals = _mm256_loadu_ps(im_buf);
                
                __m256 sq = _mm256_add_ps(_mm256_mul_ps(re_vals, re_vals), _mm256_mul_ps(im_vals, im_vals));
                
                float out_buf[8];
                _mm256_storeu_ps(out_buf, sq);
                
                for (int k = 0; k < 8; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    top_blob.row<float>(idx_j)[idx_i] = out_buf[k];
                }
            }
#endif // __AVX__
            for (; i + 3 < total; i += 4)
            {
                __m128 re_vals = _mm_setzero_ps();
                __m128 im_vals = _mm_setzero_ps();
                
                float re_buf[4], im_buf[4];
                for (int k = 0; k < 4; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    re_buf[k] = y[idx_j * frames + idx_i];
                    im_buf[k] = y[(idx_j + n_freq) * frames + idx_i];
                }
                re_vals = _mm_loadu_ps(re_buf);
                im_vals = _mm_loadu_ps(im_buf);
                
                __m128 sq = _mm_add_ps(_mm_mul_ps(re_vals, re_vals), _mm_mul_ps(im_vals, im_vals));
                
                float out_buf[4];
                _mm_storeu_ps(out_buf, sq);
                
                for (int k = 0; k < 4; k++)
                {
                    int idx_i = (i + k) % frames;
                    int idx_j = (i + k) / frames;
                    top_blob.row<float>(idx_j)[idx_i] = out_buf[k];
                }
            }
#endif // __SSE2__
            for (; i < total; i++)
            {
                int idx_i = i % frames;
                int idx_j = i / frames;
                float re = y[idx_j * frames + idx_i];
                float im = y[(idx_j + n_freq) * frames + idx_i];
                top_blob.row<float>(idx_j)[idx_i] = re * re + im * im;
            }
        }
    }

    return 0;
}

} // namespace ncnn