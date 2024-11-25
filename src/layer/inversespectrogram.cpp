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

#include "inversespectrogram.h"

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
            window_data[n_fft] = sqrt(sqsum);
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
