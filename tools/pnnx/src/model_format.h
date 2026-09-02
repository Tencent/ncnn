// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_MODEL_FORMAT_H
#define PNNX_MODEL_FORMAT_H

#include <string>

namespace pnnx {

enum ModelFormat
{
    ModelFormatUnknown,
    ModelFormatTorchScript,
    ModelFormatExportedProgramLegacy,
    ModelFormatExportedProgram
};

ModelFormat detect_model_format(const std::string& path, std::string& error);

} // namespace pnnx

#endif // PNNX_MODEL_FORMAT_H