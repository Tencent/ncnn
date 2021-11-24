

#ifndef NCNN_CONVOLUTION3D_H
#define NCNN_CONVOLUTION3D_H

#include "layer.h"

namespace ncnn {
class Convolution3D: public Layer
{
public:
    Convolution3D();

    virtual int load_param(const ParamDict &pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const;

public:
    int num_output;
    int kernel_w;
    int kernel_h;
    int kernel_d;
    int stride_w;
    int stride_h;
    int stride_d;
    int dilation_w;
    int dilation_h;
    int dilation_d;
    int bias_term;

    int weight_data_size;

    int activation_type;
    Mat activation_params;

    Mat weight_data;
    Mat bias_data;

};
}

#endif //NCNN_CONVOLUTION3D_H
