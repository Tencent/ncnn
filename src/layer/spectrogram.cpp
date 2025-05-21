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

Spectrogram::~Spectrogram()
{
    delete conv_transpose;
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
    window_data.create(n_fft);
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
            float scale = 1.f / sqrt(sqsum);

            for (int i = 0; i < n_fft; i++)
            {
                window_data[i] *= scale;
            }
        }
    }

    Mat theta;
    if (onesided)
    {
        n_freq = n_fft / 2 + 1;
    } else
    {
        n_freq = n_fft;
    }
    theta.create(n_fft,n_freq,size_t(8));

    for (int i = 0; i<n_freq; i++)
    {
        for (int j = 0; j<n_fft; j++)
        {
            theta.row<double>(i)[j] = 2 * 3.14159265358979323846 * i * j / n_fft;
        }
    }

    Mat real_basis, imag_basis;
    real_basis.create(n_fft,n_freq,size_t(8));
    imag_basis.create(n_fft,n_freq,size_t(8));

    for (int i = 0; i<n_freq; i++)
    {
        for (int j = 0; j<n_fft; j++)
        {
            real_basis.row<double>(i)[j] = cos(theta.row<double>(i)[j]);
            imag_basis.row<double>(i)[j] = -sin(theta.row<double>(i)[j]);
        }
    }

    // multiply window
    for (int i = 0; i<n_freq; i++)
    {
        for (int j = 0; j<n_fft; j++)
        {
            real_basis.row<double>(i)[j] *= window_data[j];
            imag_basis.row<double>(i)[j] *= window_data[j];
        }
    }

    if (normalized == 1)
    {
        double scale = 1.f / sqrt(n_fft);
        for (int i = 0; i<n_freq; i++)
        {
            for (int j = 0; j<n_fft; j++)
            {
                real_basis.row<double>(i)[j] *= scale;
                imag_basis.row<double>(i)[j] *= scale;
            }
        }
    }

    conv_data.create(n_fft,1,n_freq * 2);

    for (int i = 0; i<n_freq; i++)
    {
        for (int j = 0; j<n_fft; j++)
        {
            conv_data.channel(i).row<float>(0)[j]= (float)real_basis.row<double>(i)[j];
            conv_data.channel(i+n_freq).row<float>(0)[j] = (float)imag_basis.row<double>(i)[j];
        }
    }

    conv_transpose = ncnn::create_layer("Convolution1D");
    ncnn::ParamDict conv_transpose_pd;

    conv_transpose_pd.set(0,2 * n_freq); // num_output
    conv_transpose_pd.set(1,n_fft); // kernel_w
    conv_transpose_pd.set(3,hoplen); // stride_w
    conv_transpose_pd.set(19,1); // dynamic_weight

    conv_transpose->load_param(conv_transpose_pd);

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

    std::vector<Mat> inputs = {bottom_blob_bordered,conv_data};
    std::vector<Mat> outputs = {Mat()};

    Option opt_conv = opt;
    opt_conv.use_packing_layout = false;

    conv_transpose->create_pipeline(opt_conv);
    conv_transpose->forward(inputs,outputs,opt_conv);
    conv_transpose->destroy_pipeline(opt_conv);

    Mat conv_top_blob = outputs[0]; // (2 * n_freq, frames)
    float* conv_top_data = conv_top_blob;

    if (power == 0) // as complex
    {
        // copy
        for (int i = 0; i<frames; i++)
        {
            for (int j = 0; j<n_freq; j++)
            {
                top_blob.channel(j).row<float>(i)[0] = conv_top_data[j * frames + i];
                top_blob.channel(j).row<float>(i)[1] = conv_top_data[(j + n_freq) * frames + i];
            }
        }
    } else
    {
        if (power == 1) // magnitude sqrt(re * re + im * im);
        {
            // copy
            for (int i = 0; i < frames; i++)
            {
                for (int j = 0; j < n_freq; j++)
                {
                    top_blob.row<float>(j)[i] = sqrtf(conv_top_data[j * frames + i] * conv_top_data[j * frames + i] + conv_top_data[(j + n_freq) * frames + i] * conv_top_data[(j + n_freq) * frames + i]);
                }
            }
        } else if (power == 2) // power re * re + im * im;
        {
            // copy
            for (int i = 0; i < frames; i++)
            {
                for (int j = 0; j< n_freq; j++)
                {
                    top_blob.row<float>(j)[i] = conv_top_data[j * frames + i] * conv_top_data[j * frames + i] + conv_top_data[(j + n_freq) * frames + i] * conv_top_data[(j + n_freq) * frames + i];
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
