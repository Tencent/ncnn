#ifndef LAYER_LAYERNORM_X86_H
#define LAYER_LAYERNORM_X86_H

#include "layernorm.h"

namespace ncnn {

class LayerNorm_x86 : virtual public LayerNorm
{
public:
    LayerNorm_x86();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    // param
    int affine_size;
    float eps;
    int affine;

    // model
    Mat gamma_data;
    Mat beta_data;
};

} // namespace ncnn

#endif // LAYER_LAYERNORM_X86_H