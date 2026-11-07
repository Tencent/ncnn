// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SPECTROGRAM_FFT_H
#define LAYER_SPECTROGRAM_FFT_H

#include "mat.h"

#include <cmath>

namespace ncnn {

// power-of-two FFT support shared by Spectrogram and InverseSpectrogram.
//
// the real transforms use the classical half-length trick: the DFT of a
// real sequence of length n = 2m is assembled from the m-point DFT of
// the interleaved complex sequence z[r] = x[2r] + i*x[2r+1], turning the
// naive O(n^2) per-bin loop into O(n log n). split real/imag arrays,
// radix-2 decimation-in-time, one twiddle table of exp(-i*2*pi*k/n)
// generated in double precision at layer setup.

static inline int fft_is_fast(int n)
{
    return n >= 4 && (n & (n - 1)) == 0;
}

// tables of an len-point complex transform: rev holds the bit-reversal
// permutation (len ints), tw holds len/2 interleaved (re, im) pairs of
// exp(-i*2*pi*k/len)
static inline void fft_make_tables(int len, Mat& rev, Mat& tw)
{
    tw.create(len);
    if (tw.empty())
        return;
    float* p = tw;
    const double pi = 3.14159265358979323846;
    for (int k = 0; k < len / 2; k++)
    {
        const double angle = -2.0 * pi * k / len;
        p[2 * k] = (float)cos(angle);
        p[2 * k + 1] = (float)sin(angle);
    }

    rev.create(len, (size_t)sizeof(int));
    if (rev.empty())
        return;
    int* pr = (int*)rev;
    int bits = 0;
    while ((1 << bits) < len)
        bits++;
    for (int i = 0; i < len; i++)
    {
        int r = 0;
        for (int b = 0; b < bits; b++)
            r |= ((i >> b) & 1) << (bits - 1 - b);
        pr[i] = r;
    }
}

// in-place radix-2 DIT transform of split arrays; forward computes
// sum_k x[k] * exp(-i*2*pi*j*k/n), inverse the conjugate sum scaled by 1/n
static inline void fft_complex_inplace(int n, const int* rev, const float* w,
                                       float* re, float* im, int inverse)
{
    for (int i = 0; i < n; i++)
    {
        const int j = rev[i];
        if (j > i)
        {
            float t = re[i];
            re[i] = re[j];
            re[j] = t;
            t = im[i];
            im[i] = im[j];
            im[j] = t;
        }
    }

    for (int len = 2; len <= n; len <<= 1)
    {
        const int half = len >> 1;
        const int stride = n / len;
        for (int base = 0; base < n; base += len)
        {
            for (int j = 0; j < half; j++)
            {
                const float wr = w[2 * j * stride];
                const float wi = inverse ? -w[2 * j * stride + 1] : w[2 * j * stride + 1];
                const float ur = re[base + j];
                const float ui = im[base + j];
                const float vr = re[base + j + half] * wr - im[base + j + half] * wi;
                const float vi = re[base + j + half] * wi + im[base + j + half] * wr;
                re[base + j] = ur + vr;
                im[base + j] = ui + vi;
                re[base + j + half] = ur - vr;
                im[base + j + half] = ui - vi;
            }
        }
    }

    if (inverse)
    {
        const float inv_n = 1.f / n;
        for (int i = 0; i < n; i++)
        {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

// forward transform of one real frame of n (power of two, n >= 4) floats,
// window applied on the fly when win != 0; produces the n/2+1 one-sided
// bins as interleaved (re, im) pairs in out; zr / zi are work buffers of
// n/2 floats
//
// untangle, with Z = m-point DFT of z (Z[k] periodic at k = m):
//   X[k] = 1/2 (Z[k] + conj(Z[m-k])) - i/2 * W_n^k * (Z[k] - conj(Z[m-k]))
static inline void fft_real_forward(int n, const int* revh, const float* wh,
                                    const float* w, const float* x,
                                    const float* win, float* zr, float* zi,
                                    float* out)
{
    const int m = n / 2;
    if (win != 0)
    {
        for (int k = 0; k < m; k++)
        {
            zr[k] = x[2 * k] * win[2 * k];
            zi[k] = x[2 * k + 1] * win[2 * k + 1];
        }
    }
    else
    {
        for (int k = 0; k < m; k++)
        {
            zr[k] = x[2 * k];
            zi[k] = x[2 * k + 1];
        }
    }
    fft_complex_inplace(m, revh, wh, zr, zi, 0);

    for (int k = 0; k <= m; k++)
    {
        const int kc = k % m;
        const int kh = (m - k) % m;
        const float ar = 0.5f * (zr[kc] + zr[kh]);
        const float ai = 0.5f * (zi[kc] - zi[kh]);
        const float dr = zr[kc] - zr[kh];
        const float di = zi[kc] + zi[kh];

        const float wr = w[2 * k];
        const float wi = w[2 * k + 1];
        out[2 * k] = ar + 0.5f * (wr * di + wi * dr);
        out[2 * k + 1] = ai + 0.5f * (wi * di - wr * dr);
    }
}

// inverse of fft_real_forward: n/2+1 one-sided bins in -> n real floats
// out, the exact time-domain sequence the caller started from; zr / zi
// are work buffers of n/2 floats
//
// tangle: from X[k] and conj(X[m-k]) recover
//   E[k] = 1/2 (X[k] + conj(X[m-k])),  W^k O[k] = 1/2 (X[k] - conj(X[m-k]))
// and rebuild Z[k] = E[k] + i * W^-k * (W^k O[k]) of the half-length DFT
static inline void fft_real_inverse(int n, const int* revh, const float* wh,
                                    const float* w, const float* in,
                                    float* zr, float* zi, float* x)
{
    const int m = n / 2;
    for (int k = 0; k < m; k++)
    {
        const int kh = m - k;
        const float ar = 0.5f * (in[2 * k] + in[2 * kh]);
        const float ai = 0.5f * (in[2 * k + 1] - in[2 * kh + 1]);
        const float br = 0.5f * (in[2 * k] - in[2 * kh]);
        const float bi = 0.5f * (in[2 * k + 1] + in[2 * kh + 1]);

        const float wr = w[2 * k];
        const float wi = w[2 * k + 1];
        const float p = wr * br + wi * bi; // (W^k)* ... = W^-k applied
        const float q = wr * bi - wi * br;
        zr[k] = ar - q;
        zi[k] = ai + p;
    }
    fft_complex_inplace(m, revh, wh, zr, zi, 1);
    for (int k = 0; k < m; k++)
    {
        x[2 * k] = zr[k];
        x[2 * k + 1] = zi[k];
    }
}

} // namespace ncnn

#endif // LAYER_SPECTROGRAM_FFT_H
