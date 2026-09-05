// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_EXPORTED_PROGRAM_OPERATOR_H
#define PNNX_EXPORTED_PROGRAM_OPERATOR_H

#include "exported_program_schema.h"

#include <string>
#include <vector>

namespace pnnx {

struct ExportedOperatorTarget
{
    std::string namespace_name;
    std::string operator_name;
    std::string overload_name;
};

struct CanonicalExportedArgument
{
    std::string name;
    ExportedArgument value;
};

int parse_exported_operator_target(const std::string& target, ExportedOperatorTarget& result, std::string& error);

int validate_exported_program_opset(const ExportedProgramHeader& header, std::string& error);

int canonicalize_exported_arguments(const ExportedNode& node,
                                    const ExportedProgramHeader& header,
                                    ExportedOperatorTarget& target,
                                    std::vector<CanonicalExportedArgument>& result,
                                    std::string& error);

} // namespace pnnx

#endif // PNNX_EXPORTED_PROGRAM_OPERATOR_H
