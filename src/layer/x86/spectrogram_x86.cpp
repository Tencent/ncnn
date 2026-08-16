// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "spectrogram_x86.h"

#include "cpu.h"
#include "layer_type.h"

#include <math.h>

namespace ncnn {

Spectrogram_x86::Spectrogram_x86()
{
    conv1d = 0;
}

int Spectrogram_x86::load_param(const ParamDict& pd)
{
    int ret = Spectrogram::load_param(pd);
    if (ret != 0)
        return ret;

    // stft as conv1d: kernel=n_fft stride=hoplen
    // re[i] = sum_k x[k] * window[k] * cos(2*pi*i*k/n_fft) * norm
    // im[i] = sum_k x[k] * (-window[k] * sin(2*pi*i*k/n_fft)) * norm
    const int freqs_onesided = n_fft / 2 + 1;
    const int num_output = freqs_onesided * 2;

    float norm = 1.f;
    if (normalized == 1)
        norm = 1.f / sqrt((float)n_fft);
    if (normalized == 2)
        norm = window_data[n_fft];

    dft_weight.create(num_output * n_fft);
    {
        // planar: row i = re_i, row freqs_onesided + i = im_i
        float* weight = dft_weight;
        for (int i = 0; i < freqs_onesided; i++)
        {
            float* re_row = weight + i * n_fft;
            float* im_row = weight + (freqs_onesided + i) * n_fft;
            for (int k = 0; k < n_fft; k++)
            {
                double angle = 2 * 3.14159265358979323846 * i * k / n_fft;
                re_row[k] = (float)(window_data[k] * cos(angle) * norm);
                im_row[k] = (float)(-window_data[k] * sin(angle) * norm);
            }
        }
    }

    conv1d = create_layer_cpu(LayerType::Convolution1D);
    if (!conv1d)
        return -100;
    {
        ParamDict pd_conv;
        pd_conv.set(0, num_output);         // num_output
        pd_conv.set(1, n_fft);              // kernel_w
        pd_conv.set(2, 1);                  // dilation_w
        pd_conv.set(3, hoplen);             // stride_w
        pd_conv.set(4, 0);                  // pad_left
        pd_conv.set(15, 0);                 // pad_right
        pd_conv.set(5, 0);                  // bias_term
        pd_conv.set(6, num_output * n_fft); // weight_data_size
        pd_conv.set(9, 0);                  // activation_type
        conv1d->load_param(pd_conv);

        Mat weights[1] = {dft_weight};
        conv1d->load_model(ModelBinFromMatArray(weights));
    }

    return 0;
}

int Spectrogram_x86::create_pipeline(const Option& opt)
{
    if (conv1d)
        conv1d->create_pipeline(opt);

    return 0;
}

int Spectrogram_x86::destroy_pipeline(const Option& opt)
{
    if (conv1d)
    {
        conv1d->destroy_pipeline(opt);
        delete conv1d;
        conv1d = 0;
    }

    return 0;
}

int Spectrogram_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // https://pytorch.org/audio/stable/generated/torchaudio.functional.spectrogram.html

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

    // conv1d: [w=frames, h=2*freqs_onesided]
    Mat conv_out;
    {
        Mat conv_out_packed;
        int ret = conv1d->forward(bottom_blob_bordered, conv_out_packed, opt);
        if (ret != 0)
            return ret;

        // x86 conv1d may produce pack4/8/16 output, unpack to pack1
        if (conv_out_packed.elempack != 1)
        {
            convert_packing(conv_out_packed, conv_out, 1, opt);
        }
        else
        {
            conv_out = conv_out_packed;
        }
    }

    const int frames = conv_out.w;
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
        const float* re_ptr = conv_out.row(i);
        const float* im_ptr = conv_out.row(freqs_onesided + i);
        float* outptr = power == 0 ? top_blob.channel(i) : top_blob.row(i);

        if (power == 0)
        {
            for (int j = 0; j < frames; j++)
            {
                outptr[0] = re_ptr[j];
                outptr[1] = im_ptr[j];
                outptr += 2;
            }
        }
        if (power == 1)
        {
            for (int j = 0; j < frames; j++)
            {
                outptr[j] = sqrt(re_ptr[j] * re_ptr[j] + im_ptr[j] * im_ptr[j]);
            }
        }
        if (power == 2)
        {
            for (int j = 0; j < frames; j++)
            {
                outptr[j] = re_ptr[j] * re_ptr[j] + im_ptr[j] * im_ptr[j];
            }
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
