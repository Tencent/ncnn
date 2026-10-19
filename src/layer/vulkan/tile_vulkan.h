// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_TILE_VULKAN_H
#define LAYER_TILE_VULKAN_H

#include "tile.h"

namespace ncnn {

class Tile_vulkan : public Tile
{
public:
    Tile_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Tile::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_tile;
    Pipeline* pipeline_tile_pack4;
};

} // namespace ncnn

#endif // LAYER_TILE_VULKAN_H
