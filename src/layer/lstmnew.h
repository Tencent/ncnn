#ifndef LAYER_LSTMNEW_H
#define LAYER_LSTMNEW_H

#include "layer.h"

namespace ncnn {

class LstmNew : public Layer
{
public:
    LstmNew();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    //virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    // param
    int num_output;
    int weight_data_size;

    // model
    Mat weight_i_data;
    Mat weight_h_data;
    Mat bias_c_data;
};

} // namespace ncnn

#endif // LAYER_LSTMNEW_H