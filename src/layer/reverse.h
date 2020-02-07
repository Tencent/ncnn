#ifndef LAYER_REVERSE_H
#define LAYER_REVERSE_H

#include "layer.h"

namespace ncnn {

class Reverse : public Layer
{
public:
    Reverse();


    // virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;


};

} // namespace ncnn

#endif // LAYER_REVERSE_H
