// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_MODEL_STAT_H
#define PNNX_MODEL_STAT_H

#include <stdint.h>

#include <string>

namespace pnnx {

class Graph;

struct ModelStat
{
    ModelStat()
        : flops(0), memops(0)
    {
    }

    uint64_t flops;
    uint64_t memops;
};

ModelStat get_model_stat(const Graph& graph);
std::string format_model_stat_input_shapes(const Graph& graph);
std::string format_model_stat_ops(uint64_t ops);

} // namespace pnnx

#endif // PNNX_MODEL_STAT_H
