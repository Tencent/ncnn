// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_pt2.h"

#include <stdio.h>

#include "pt2_archive.h"
#include "pt2_graph_lowering.h"
#include "pt2_program.h"
#include "pt2_weights.h"

namespace pnnx {

int load_pt2(const std::string& path, Graph& graph)
{
    Pt2ArchiveReader archive;
    if (archive.open(path) != 0)
    {
        fprintf(stderr, "load pt2 archive failed: %s\n", archive.error.c_str());
        return -1;
    }

    Pt2Program program;
    if (load_pt2_program(archive, program) != 0)
    {
        fprintf(stderr, "load pt2 program failed: %s\n", program.error.c_str());
        return -1;
    }

    fprintf(stderr, "pt2 container=%s", archive.container_kind == Pt2ContainerArchive ? "archive" : "legacy-exported-program");
    if (!archive.archive_version.empty())
        fprintf(stderr, " archive_version=%s", archive.archive_version.c_str());
    fprintf(stderr, " schema=%d.%d opset=", program.schema_major, program.schema_minor);
    for (std::map<std::string, int>::const_iterator it = program.opset_versions.begin(); it != program.opset_versions.end(); ++it)
    {
        if (it != program.opset_versions.begin())
            fprintf(stderr, ",");
        fprintf(stderr, "%s:%d", it->first.c_str(), it->second);
    }
    fprintf(stderr, " producer=torch%s%s\n", program.torch_version.empty() ? "" : "-", program.torch_version.c_str());

    Pt2Weights weights;
    if (load_pt2_weights(archive, program, weights) != 0)
    {
        fprintf(stderr, "load pt2 weights failed: %s\n", weights.error.c_str());
        return -1;
    }

    std::string error;
    if (lower_pt2_graph(program, weights, graph, error) != 0)
    {
        fprintf(stderr, "lower pt2 graph failed: %s\n", error.c_str());
        return -1;
    }

    return 0;
}

} // namespace pnnx
