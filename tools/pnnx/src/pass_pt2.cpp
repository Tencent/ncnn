// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_pt2.h"

#include "pass_pt2/normalize_arguments.h"
#include "pass_pt2/normalize_as_strided.h"
#include "pass_pt2/normalize_dtype.h"
#include "pass_pt2/normalize_mutation.h"
#include "pass_pt2/normalize_scalar.h"

#include <stdio.h>

namespace pnnx {

int pass_pt2(Graph& graph, const std::map<Operator*, Pt2TensorMeta>& layout_sources)
{
    // Layout information belongs to the serialized graph. Normalize operations
    // that depend on it before the shared passes can materialize tensor values.
    for (Operator* op : graph.ops)
    {
        const auto source = layout_sources.find(op);
        if (source == layout_sources.end() || op->type == "aten::as_strided")
            continue;
        std::string error;
        if (!normalize_pt2_layout_query(op, source->second, error))
        {
            fprintf(stderr, "pt2 layout: %s at %s\n", error.c_str(), op->name.c_str());
            return -1;
        }
    }
    for (Operator* op : std::vector<Operator*>(graph.ops))
    {
        const auto source = layout_sources.find(op);
        if (source == layout_sources.end() || op->type != "aten::as_strided")
            continue;
        std::string error;
        if (!normalize_pt2_as_strided(graph, op, source->second, error))
        {
            fprintf(stderr, "pt2 layout: %s at %s\n", error.c_str(), op->name.c_str());
            return -1;
        }
    }
    normalize_pt2_arguments(graph);
    normalize_pt2_scalar(graph);
    if (normalize_pt2_dtype(graph) != 0)
        return -1;
    return normalize_pt2_mutation(graph);
}

} // namespace pnnx
