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
    dilation_w = pd.get(7, 1);
    dilation_h = pd.get(8, dilation_w);
    dilation_d = pd.get(9, dilation_w);
    weight_data_size = pd.get(10, 0);
    bias_term = pd.get(11, 0);
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

    // Compute dilated kernel size. This extension is achieved by calculating offsets on original
    // input, instead of extending kernel itself
    const int kernel_extend_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extend_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extend_d = dilation_d * (kernel_d - 1) + 1;

    // pad bottom blob
    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    // update shape info
    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;
    d = bottom_blob_bordered.d;

    // compute output shape
    int outw = (w - kernel_extend_w) / stride_w + 1;
    int outh = (h - kernel_extend_h) / stride_h + 1;
    int outd = (d - kernel_extend_d) / stride_d + 1;

    // compute kernel size
    const int maxk = kernel_w * kernel_h * kernel_d;


    // compute offset to align original input and kernel data
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];

    {
        int p1 = 0;
        int p2 = 0;
        int offset0 = dilation_d;
        int offset1 = d * dilation_h - kernel_d * dilation_d;
        int offset2 = (h*d) * dilation_w - h * kernel_h * dilation_h - kernel_h * dilation_h;
        for(int i = 0; i < kernel_w; ++i) {
            for(int j = 0; j < kernel_h; ++j) {
                for(int k = 0; k < kernel_d; ++k) {
                    space_ofs[p1] = p2;
                    p1++;
                    p2 += offset0;
                }
                p2 += offset1;
            }
            p2 += offset2;
        }
    }

    // create top blob
    top_blob.create(outw, outh, outd, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // main loop
    // #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int j = 0; j < outw; j++)
        {
            for (int i = 0; i < outh; i++)
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
                        // (w*d): offset when you go across one h
                        // (d): offset when you go across one w
                        const float* sptr = (float*)m.data + (h*d) * j * stride_w + (d) * i * stride_h + k * stride_d;

                        for (int l = 0; l < maxk; l++)
                        {
                            float val = sptr[space_ofs[l]];
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

                // move forward output pointer
                outptr += outd;
            }
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


