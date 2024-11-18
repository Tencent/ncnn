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
    power = pd.get(1, 0);
    hoplen = pd.get(2, n_fft / 4);
    winlen = pd.get(3, n_fft);

    return 0;
}

int InverseSpectrogram::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py#L630

    // TODO custom window
    // TODO padding for center=True
    // TODO onesided=True

    const int frames = bottom_blob.h;
    // const int freqs = bottom_blob.c;
    // assert freqs == n_fft or freqs == n_fft / 2 + 1

    const int outsize = (frames - 1) * hoplen + (n_fft - n_fft / 2 * 2); //  center=1
    // const int outsize = (frames - 1) * hoplen + n_fft; //  center=0

    const size_t elemsize = bottom_blob.elemsize;

    top_blob.create(outsize, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    Mat top_blob_padded(outsize + n_fft, elemsize, opt.workspace_allocator);
    Mat window_sumsquare(outsize + n_fft, elemsize, opt.workspace_allocator);
    for (int i = 0; i < outsize; i++)
    {
        top_blob_padded[i] = 0.f;
        window_sumsquare[i] = 0.f;
    }

    for (int j = 0; j < frames; j++)
    {
        // collect complex
        Mat sp(2, n_fft);
        for (int k = 0; k < n_fft; k++)
        {
            sp.row(k)[0] = bottom_blob.channel(k).row(j)[0];
            sp.row(k)[1] = bottom_blob.channel(k).row(j)[1];
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < n_fft; i++)
        {
            // inverse dft
            float re = 0.f;
            // float im = 0.f;
            for (int k = 0; k < n_fft; k++)
            {
                double angle = 2 * M_PI * i * k / n_fft;

                re += sp.row(k)[0] * cos(angle) - sp.row(k)[1] * sin(angle);
                // im += sp.row(k)[0] * sin(angle) + sp.row(k)[1] * cos(angle);
            }

            re /= n_fft;
            // im /= n_fft;

            // apply hann window
            re *= 0.5f * (1 - cos(2 * M_PI * i / n_fft));

            // square hann window
            window_sumsquare[j * hoplen + i] += (0.5f * (1 - cos(2 * M_PI * i / n_fft))) * (0.5f * (1 - cos(2 * M_PI * i / n_fft)));

            top_blob_padded[j * hoplen + i] += re;
        }
    }

    // cut padding
    for (int i = 0; i < outsize; i++)
    {
        top_blob[i] = top_blob_padded[n_fft / 2 + i] / window_sumsquare[n_fft / 2 + i];
    }

    return 0;
}

} // namespace ncnn
