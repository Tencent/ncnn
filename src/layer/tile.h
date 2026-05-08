// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_TILE_H
#define LAYER_TILE_H

#include "layer.h"

namespace ncnn {

class Tile : public Layer
{
public:
    Tile();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int axis;
    int tiles;
    Mat repeats;
};

} // namespace ncnn

#endif // LAYER_TILE_H
