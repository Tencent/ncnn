// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_TILE_POWERPC_H
#define LAYER_TILE_POWERPC_H

#include "tile.h"

namespace ncnn {

class Tile_powerpc : public Tile
{
public:
    Tile_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_TILE_POWERPC_H
