//
// Created by RogerOu on 2018/9/28.
//

#ifndef LAYER_CLIP_ARM_H
#define LAYER_CLIP_ARM_H

#include "layer.h"
#include "clip.h"

namespace ncnn {
    class Clip_arm : public Clip {
        
        virtual int forward_inplace(Mat &bottom_top_blob, const Option &opt) const;

    };
}
#endif //LAYER_CLIP_ARM_H
