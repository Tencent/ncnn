// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "spectrogram.h"

namespace ncnn {

Spectrogram::Spectrogram()
{
    one_blob_only = true;
    support_inplace = false;
}

int Spectrogram::load_param(const ParamDict& pd)
{
    n_fft = pd.get(0, 0);
    power = pd.get(1, 0);
    hoplen = pd.get(2, n_fft / 4);
    winlen = pd.get(3, n_fft);
    window_type = pd.get(4, 0);
    center = pd.get(5, 1);
    pad_type = pd.get(6, 2);
    normalized = pd.get(7, 0);
    onesided = pd.get(8, 1);

    // assert winlen <= n_fft
    // generate window
    window_data.create(normalized == 2 ? n_fft + 1 : n_fft);
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
            window_data[n_fft] = 1.f / sqrt(sqsum);
        }
    }

    return 0;
}

int Spectrogram::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // https://pytorch.org/audio/stable/generated/torchaudio.functional.spectrogram.html

    // TODO custom window

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

    // const int frames = size / hoplen + 1;
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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < freqs_onesided; i++)
    {
        const float* ptr = bottom_blob_bordered;
        float* outptr = power == 0 ? top_blob.channel(i) : top_blob.row(i);

        for (int j = 0; j < frames; j++)
        {
            float re = 0.f;
            float im = 0.f;
            for (int k = 0; k < n_fft; k++)
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
                float norm = 1.f / sqrt(n_fft);
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
                outptr[0] = sqrt(re * re + im * im);
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

} // namespace ncnn
