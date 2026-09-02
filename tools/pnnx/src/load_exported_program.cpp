// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exported_program.h"

#include <stdio.h>

#include "exported_program.h"

namespace pnnx {

int load_exported_program(const std::string& path, Graph& /*graph*/)
{
    pt2::ExportedProgramArchive archive;
    std::string error;
    if (!pt2::load_exported_program_archive(path, archive, error))
    {
        fprintf(stderr, "load exported program failed: %s\n", error.c_str());
        return -1;
    }

    fprintf(stderr, "load exported program failed: graph import is not supported yet\n");
    return -1;
}

} // namespace pnnx