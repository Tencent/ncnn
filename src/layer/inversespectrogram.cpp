// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "inversespectrogram.h"

#include "spectrogram_fft.h"

#include "cpu.h"

#include <limits.h>
#include <string.h>

namespace ncnn {

InverseSpectrogram::InverseSpectrogram()
{
    one_blob_only = true;
    support_inplace = false;
}

int InverseSpectrogram::load_param(const ParamDict& pd)
{
    n_fft = pd.get(0, 0);
    returns = pd.get(1, 0);
    hoplen = pd.get(2, n_fft / 4);
    winlen = pd.get(3, n_fft);
    window_type = pd.get(4, 0);
    center = pd.get(5, 1);
    normalized = pd.get(7, 0);

#if NCNN_VALIDATION
    if (returns < 0 || returns > 2)
        return -1;

    if (n_fft <= 0 || winlen <= 0 || winlen > n_fft)
        return -1;

    if (hoplen <= 0 || window_type < 0 || window_type > 2 || normalized < 0 || normalized > 2 || (normalized == 2 && n_fft == INT_MAX))
        return -1;

    // leave room for Mat alignment, the reference count and fastMalloc overhead
    const size_t max_window_size = ((size_t)-1 - 15 - sizeof(int) - sizeof(void*) - NCNN_MALLOC_ALIGN - NCNN_MALLOC_OVERREAD) / sizeof(float);
    if ((size_t)n_fft + (normalized == 2 ? 1 : 0) > max_window_size)
        return -1;
#endif // NCNN_VALIDATION

    // generate window
    window_data.create(normalized == 2 ? n_fft + 1 : n_fft);
    if (window_data.empty())
        return -100;
    {
        float* p = window_data;
        for (int i = 0; i < (n_fft - winlen) / 2; i++)
        {
            *p++ = 0.f;
        }
        if (window_type == 0)
        {
            // all ones
            for (int i = 0; i < winlen; i++)
            {
                *p++ = 1.f;
            }
        }
        if (window_type == 1)
        {
            // hann window
            for (int i = 0; i < winlen; i++)
            {
                *p++ = 0.5f * (1 - cosf(2 * 3.14159265358979323846 * i / winlen));
            }
        }
        if (window_type == 2)
        {
            // hamming window
            for (int i = 0; i < winlen; i++)
            {
                *p++ = 0.54f - 0.46f * cosf(2 * 3.14159265358979323846 * i / winlen);
            }
        }
        for (int i = 0; i < n_fft - winlen - (n_fft - winlen) / 2; i++)
        {
            *p++ = 0.f;
        }

        // pre-calculated window norm factor
        if (normalized == 2)
        {
            float sqsum = 0.f;
            for (int i = 0; i < n_fft; i++)
            {
                sqsum += window_data[i] * window_data[i];
            }
            window_data[n_fft] = sqrt(sqsum);
        }
    }

    // power-of-two fast path: precompute the half-length transform tables
    if (fft_is_fast(n_fft))
    {
        fft_make_tables(n_fft / 2, fft_rev, fft_tw);
        fft_tw_real.create(n_fft + 2);
        if (fft_rev.empty() || fft_tw.empty() || fft_tw_real.empty())
            return -100;
        float* p = fft_tw_real;
        const double pi = 3.14159265358979323846;
        for (int k = 0; k <= n_fft / 2; k++)
        {
            const double angle = -2.0 * pi * k / n_fft;
            p[2 * k] = (float)cos(angle);
            p[2 * k + 1] = (float)sin(angle);
        }
    }

    return 0;
}

int InverseSpectrogram::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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

    if (fft_is_fast(n_fft) && onesided == 1)
    {
        // O(n log n) per frame: tangle the one-sided spectrum into the
        // half-length spectrum, inverse transform, overlap-add.
        //
        // frames are processed in bounded chunks; each chunk runs the
        // conventional omp-parallel-for over its frames, accumulating
        // into per-thread buffers that only span the chunk's output
        // range (a few MB, independent of the signal length), then the
        // per-thread buffers merge serially into top_blob and
        // window_sumsquare (the overlap-add is associative). every
        // slot is zero-initialized before the team runs, so a team
        // smaller than the requested thread count (no OpenMP build,
        // OMP_THREAD_LIMIT, nested parallelism) stays exact.
        //
        // the paired bins 1..m-1 are Hermitian and reconstruct exactly
        // through the half-length transform. the endpoint bins k = 0
        // and k = m are not a Hermitian pair for arbitrary complex
        // input, so their imaginary parts are carried separately:
        //   y_im[n] = (Im X[0] + (-1)^n Im X[m]) / n_fft
        // and the real transform runs on the endpoint bins with their
        // real parts only.
        const int m = n_fft / 2;
        const int off = center == 1 ? n_fft / 2 : 0;
        int nbuffers = opt.num_threads < 1 ? 1 : opt.num_threads;

        // chunk size: per-thread accumulators stay around 1 MB
        const int chunk_frames = 1048576 / hoplen > 1 ? 1048576 / hoplen : 1;
        // chunk output span bound
        int maxspan = chunk_frames * hoplen + n_fft;
        if (maxspan > outsize)
            maxspan = outsize;
        const size_t valwmax = returns == 0 ? (size_t)2 * maxspan : (size_t)maxspan;
        const size_t tstride = (size_t)(n_fft + 2) + m + m + n_fft + valwmax + maxspan;
        Mat scratch(tstride * nbuffers, (size_t)4u, opt.workspace_allocator);
        if (scratch.empty())
            return -100;
        memset((float*)scratch, 0, tstride * nbuffers * sizeof(float));

        const float norm1 = sqrt((float)n_fft);
        const float norm2 = normalized == 2 ? window_data[n_fft] : 0.f;
        const float inv_nfft = 1.f / n_fft;
        const float* win = window_data;
        const int* revh = (const int*)fft_rev;
        const float* wh = fft_tw;
        const float* wf = fft_tw_real;
        float* top = (float*)top_blob;
        float* wss = (float*)window_sumsquare;

        for (int f0 = 0; f0 < frames; f0 += chunk_frames)
        {
            const int f1 = f0 + chunk_frames < frames ? f0 + chunk_frames : frames;
            const int nf = f1 - f0;
            int nth = nbuffers < nf ? nbuffers : nf;

            // this chunk's output span
            int s0 = f0 * hoplen - off;
            if (s0 < 0)
                s0 = 0;
            int s1 = (f1 - 1) * hoplen - off + n_fft;
            if (s1 > outsize)
                s1 = outsize;
            if (s1 <= s0)
                continue;
            const int span = s1 - s0;
            const size_t valw = returns == 0 ? (size_t)2 * span : (size_t)span;

            #pragma omp parallel for num_threads(nth) schedule(static)
            for (int jj = 0; jj < nf; jj++)
            {
                const int tid = get_omp_thread_num();
                float* xin = (float*)scratch + (size_t)tid * tstride;
                float* zr = xin + (n_fft + 2);
                float* zi = zr + m;
                float* xr = zi + m;
                float* val = xr + n_fft;
                float* sq = val + valwmax;
                const int j = f0 + jj;

                // collect the one-sided spectrum; fft_real_inverse
                // tangles the Hermitian mirror internally, so bins
                // m+1..n-1 are never materialized
                for (int k = 0; k <= m; k++)
                {
                    xin[2 * k] = bottom_blob.channel(k).row(j)[0];
                    xin[2 * k + 1] = bottom_blob.channel(k).row(j)[1];
                }

                if (normalized == 1)
                {
                    for (int i = 0; i < 2 * (m + 1); i++)
                    {
                        xin[i] *= norm1;
                    }
                }
                if (normalized == 2)
                {
                    for (int i = 0; i < 2 * (m + 1); i++)
                    {
                        xin[i] *= norm2;
                    }
                }

                // split off the non-Hermitian endpoint imaginary parts
                const float imx0 = xin[1];
                const float imxm = xin[2 * m + 1];
                xin[1] = 0.f;
                xin[2 * m + 1] = 0.f;

                fft_real_inverse(n_fft, revh, wh, wf, xin, zr, zi, xr);

                for (int i = 0; i < n_fft; i++)
                {
                    const int output_index = j * hoplen + i - off;
                    if (output_index < s0 || output_index >= s1)
                        continue;
                    const int si = output_index - s0;

                    const float wv = win[i];
                    const float vre = xr[i] * wv;
                    const float vim = (imx0 + ((i & 1) ? -imxm : imxm)) * inv_nfft * wv;

                    // square window
                    sq[si] += wv * wv;

                    if (returns == 0)
                    {
                        val[2 * si] += vre;
                        val[2 * si + 1] += vim;
                    }
                    if (returns == 1)
                    {
                        val[si] += vre;
                    }
                    if (returns == 2)
                    {
                        val[si] += vim;
                    }
                }
            }

            // merge this chunk's per-thread accumulators (nth slots,
            // all zero-initialized above, exact for any team size)
            for (int t = 0; t < nth; t++)
            {
                const float* val = (float*)scratch + (size_t)t * tstride + (n_fft + 2) + m + m + n_fft;
                const float* sq = val + valwmax;
                if (returns == 0)
                {
                    for (int i = 0; i < span; i++)
                    {
                        top[2 * (s0 + i)] += val[2 * i];
                        top[2 * (s0 + i) + 1] += val[2 * i + 1];
                    }
                }
                else
                {
                    for (int i = 0; i < span; i++)
                    {
                        top[s0 + i] += val[i];
                    }
                }
                for (int i = 0; i < span; i++)
                {
                    wss[s0 + i] += sq[i];
                }
            }

            // re-zero the used slots for the next chunk
            memset((float*)scratch, 0, tstride * nth * sizeof(float));
        }
    }
    else
    {
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

            if (normalized == 1)
            {
                float norm = sqrt(n_fft);
                for (int i = 0; i < 2 * n_fft; i++)
                {
                    sp[i] *= norm;
                }
            }
            if (normalized == 2)
            {
                float norm = window_data[n_fft];
                for (int i = 0; i < 2 * n_fft; i++)
                {
                    sp[i] *= norm;
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < n_fft; i++)
            {
                // inverse dft
                float re = 0.f;
                float im = 0.f;
                for (int k = 0; k < n_fft; k++)
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

} // namespace ncnn
