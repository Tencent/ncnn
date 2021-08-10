//
// Created by 余浩文 on 2021/8/6.
//

#include "convolution3d.h"
namespace ncnn {

Convolution3D::Convolution3D()
{
    one_blob_only = true;
    support_inplace = false;
}

int Convolution3D::load_param(const ParamDict& pd)
{
    num_output = pd.get(0,0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(2, kernel_w);
    kernel_d = pd.get(3, kernel_w);
    stride_w = pd.get(4, 1);
    stride_h = pd.get(5, stride_w);
    stride_d = pd.get(6, stride_w);
    weight_data_size = pd.get(7, 0);
    bias_term = pd.get(8, 0);
    return 0;
}

int Convolution3D::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int Convolution3D::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // record shape info
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    // pad bottom blob
    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    // compute output shape
    int outw = w / stride_w + 1;
    int outh = h / stride_h + 1;
    int outd = d / stride_d + 1;

    // compute kernel size
    const int maxk = kernel_w * kernel_h * kernel_d;

    // create top blob
    top_blob.create(outw, outh, outd, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // main loop
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                for (int k = 0; k < outd; k++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[p];

                    const float* kptr = (const float*)weight_data + maxk * channels * p;

                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob_bordered.channel(q);
                        const float* sptr = m.plane(k * stride_d) + i * stride_w + j * stride_h;

                        for (int l = 0; l < maxk; l++)
                        {
                            float val = sptr[l];
                            float wt = kptr[l];
                            sum += val * wt;
                        }

                        kptr += maxk;
                    }

                    if (activation_type == 1)
                    {
                        sum = std::max(sum, 0.f);
                    }
                    else if (activation_type == 2)
                    {
                        float slope = activation_params[0];
                        sum = sum > 0.f ? sum : sum * slope;
                    }
                    else if (activation_type == 3)
                    {
                        float min = activation_params[0];
                        float max = activation_params[1];
                        if (sum < min)
                            sum = min;
                        if (sum > max)
                            sum = max;
                    }
                    else if (activation_type == 4)
                    {
                        sum = static_cast<float>(1.f / (1.f + exp(-sum)));
                    }
                    else if (activation_type == 5)
                    {
                        const float MISH_THRESHOLD = 20;
                        float x = sum, y;
                        if (x > MISH_THRESHOLD)
                            y = x;
                        else if (x < -MISH_THRESHOLD)
                            y = expf(x);
                        else
                            y = logf(expf(x) + 1);
                        sum = static_cast<float>(x * tanh(y));
                    }

                    outptr[k] = sum;
                }

                outptr += outd;
            }

            outptr += outw * outd;
        }
    }

    return 0;
}

void Convolution3D::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    bottom_blob_bordered = bottom_blob;
    return;
}

}


