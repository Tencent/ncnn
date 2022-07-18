#include "layernorm_x86.h"

#include <math.h>

namespace ncnn {

LayerNorm_x86::LayerNorm_x86()
    : LayerNorm()
{
}

int LayerNorm_x86::load_param(const ParamDict& pd)
{
    return LayerNorm::load_param(pd);
}

int LayerNorm_x86::load_model(const ModelBin& mb)
{
    return LayerNorm::load_model(mb);
}

int LayerNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    return LayerNorm::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
