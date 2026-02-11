// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CUMULATIVESUM_VULKAN_H
#define LAYER_CUMULATIVESUM_VULKAN_H

#include "cumulativesum.h"

namespace ncnn {

class CumulativeSum_vulkan : public CumulativeSum
{
public:
    CumulativeSum_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using CumulativeSum::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_cumulativesum_blockscan;
    Pipeline* pipeline_cumulativesum_blocksums_scan;
    Pipeline* pipeline_cumulativesum_addoffset;
};

} // namespace ncnn

#endif // LAYER_CUMULATIVESUM_VULKAN_H
