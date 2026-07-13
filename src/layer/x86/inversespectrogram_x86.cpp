// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "inversespectrogram_x86.h"

#include "cpu.h"
#include "layer_type.h"

#include <math.h>

namespace ncnn {

InverseSpectrogram_x86::InverseSpectrogram_x86()
{
    conv1d = 0;
}

int InverseSpectrogram_x86::load_param(const ParamDict& pd)
{
    int ret = InverseSpectrogram::load_param(pd);
    if (ret != 0)
        return ret;

    // istft per-frame idft: tap m of frame j gets
    // re += (sp_re[k] * cos(2*pi*k*m/n_fft) - sp_im[k] * sin(2*pi*k*m/n_fft)) / n_fft * window[m] * norm
    // im += (sp_re[k] * sin(2*pi*k*m/n_fft) + sp_im[k] * cos(2*pi*k*m/n_fft)) / n_fft * window[m] * norm
    // as conv1d 1x1: num_input = 2*n_fft (sp_re then sp_im), num_output = 2*n_fft (re taps then im taps)
    const int num_input = n_fft * 2;
    const int num_output = n_fft * 2;

    float norm = 1.f;
    if (normalized == 1)
        norm = sqrt((float)n_fft);
    if (normalized == 2)
        norm = window_data[n_fft];

    // conv1d weight layout: [out_ch][in_ch q], kernel=1
    // out_ch [0, n_fft) = re taps, out_ch [n_fft, 2*n_fft) = im taps
    // in_ch [0, n_fft) = sp_re, in_ch [n_fft, 2*n_fft) = sp_im
    idft_weight.create(num_output * num_input);
    {
        float* weight = idft_weight;
        for (int m = 0; m < n_fft; m++)
        {
            float* re_row = weight + m * num_input;
            float* im_row = weight + (n_fft + m) * num_input;
            for (int k = 0; k < n_fft; k++)
            {
                double angle = 2 * 3.14159265358979323846 * k * m / n_fft;
                re_row[k] = (float)(window_data[m] * cos(angle) / n_fft * norm);
                re_row[n_fft + k] = (float)(-window_data[m] * sin(angle) / n_fft * norm);
                im_row[k] = (float)(window_data[m] * sin(angle) / n_fft * norm);
                im_row[n_fft + k] = (float)(window_data[m] * cos(angle) / n_fft * norm);
            }
        }
    }

    conv1d = create_layer_cpu(LayerType::Convolution1D);
    {
        ParamDict pd_conv;
        pd_conv.set(0, num_output);             // num_output
        pd_conv.set(1, 1);                      // kernel_w
        pd_conv.set(2, 1);                      // dilation_w
        pd_conv.set(3, 1);                      // stride_w
        pd_conv.set(4, 0);                      // pad_left
        pd_conv.set(15, 0);                     // pad_right
        pd_conv.set(5, 0);                      // bias_term
        pd_conv.set(6, num_output * num_input); // weight_data_size
        pd_conv.set(9, 0);                      // activation_type
        conv1d->load_param(pd_conv);

        Mat weights[1] = {idft_weight};
        conv1d->load_model(ModelBinFromMatArray(weights));
    }

    return 0;
}

int InverseSpectrogram_x86::create_pipeline(const Option& opt)
{
    if (conv1d)
        conv1d->create_pipeline(opt);

    return 0;
}

int InverseSpectrogram_x86::destroy_pipeline(const Option& opt)
{
    if (conv1d)
    {
        conv1d->destroy_pipeline(opt);
        delete conv1d;
        conv1d = 0;
    }

    return 0;
}

int InverseSpectrogram_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py#L630

    const int frames = bottom_blob.h;
    const int freqs = bottom_blob.c;
    // assert freqs == n_fft or freqs == n_fft / 2 + 1

    const int onesided = freqs == n_fft / 2 + 1 ? 1 : 0;

    const size_t elemsize = bottom_blob.elemsize;

    // collect full complex spectrum as conv input [w=frames, h=2*n_fft]
    // row k = re of bin k, row n_fft + k = im of bin k
    Mat sp(frames, n_fft * 2, elemsize, opt.workspace_allocator);
    if (sp.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int k = 0; k < n_fft; k++)
    {
        float* re_row = sp.row(k);
        float* im_row = sp.row(n_fft + k);
        if (k < freqs)
        {
            const float* ptr = bottom_blob.channel(k);
            for (int j = 0; j < frames; j++)
            {
                re_row[j] = ptr[0];
                im_row[j] = ptr[1];
                ptr += 2;
            }
        }
        else // if (onesided)
        {
            // conjugate mirror of bin n_fft - k
            const float* ptr = bottom_blob.channel(n_fft - k);
            for (int j = 0; j < frames; j++)
            {
                re_row[j] = ptr[0];
                im_row[j] = -ptr[1];
                ptr += 2;
            }
        }
    }

    // conv1d 1x1: [w=frames, h=n_fft], row m = idft tap m over frames
    Mat conv_out;
    {
        Mat conv_out_packed;
        int ret = conv1d->forward(sp, conv_out_packed, opt);
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

    const int pad = center == 1 ? n_fft / 2 : 0;
    const int outsize = (frames - 1) * hoplen + n_fft - pad * 2;

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

    // overlap-add shifted taps into uncropped buffers, with square window norm
    // (same accumulation as reference)
    const int full_size = outsize + pad * 2;
    Mat yfull_re(full_size, elemsize, opt.workspace_allocator);
    Mat yfull_im(full_size, elemsize, opt.workspace_allocator);
    Mat window_sumsquare(full_size, elemsize, opt.workspace_allocator);
    if (yfull_re.empty() || yfull_im.empty() || window_sumsquare.empty())
        return -100;
    yfull_re.fill(0.f);
    yfull_im.fill(0.f);
    window_sumsquare.fill(0.f);

    for (int m = 0; m < n_fft; m++)
    {
        const float* re_row = conv_out.row(m);
        const float* im_row = conv_out.row(n_fft + m);
        const float w2 = window_data[m] * window_data[m];
        for (int j = 0; j < frames; j++)
        {
            const int idx = j * hoplen + m;
            yfull_re[idx] += re_row[j];
            yfull_im[idx] += im_row[j];
            window_sumsquare[idx] += w2;
        }
    }

    // crop center pad and normalize
    if (returns == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < outsize; i++)
        {
            float* outptr = top_blob.row(i);
            const float ws = window_sumsquare[i + pad];
            const float re = yfull_re[i + pad];
            const float im = yfull_im[i + pad];
            if (ws != 0.f)
            {
                outptr[0] = re / ws;
                outptr[1] = im / ws;
            }
            else
            {
                outptr[0] = re;
                outptr[1] = im;
            }
        }
    }
    else
    {
        const Mat& yfull = returns == 2 ? yfull_im : yfull_re;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < outsize; i++)
        {
            const float ws = window_sumsquare[i + pad];
            top_blob[i] = ws != 0.f ? yfull[i + pad] / ws : yfull[i + pad];
        }
    }

    return 0;
}

} // namespace ncnn
