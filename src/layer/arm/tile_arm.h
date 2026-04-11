// ARM NEON header for Tile
// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_TILE_ARM_H
#define LAYER_TILE_ARM_H

#include "tile.h"

namespace ncnn {

class Tile_arm : public virtual Tile
{
public:
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_TILE_ARM_H
