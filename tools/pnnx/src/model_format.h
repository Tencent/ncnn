// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_MODEL_FORMAT_H
#define PNNX_MODEL_FORMAT_H

#include <string>

namespace pnnx {

enum ModelFormat
{
    ModelFormatOther,
    ModelFormatTorchScript,
    ModelFormatPt2LegacyExportedProgram,
    ModelFormatPt2Archive,
    ModelFormatUnknownZip
};

struct ModelFormatInfo
{
    ModelFormat format;
    std::string archive_version;
    std::string diagnostic;
};

int probe_model_format(const std::string& path, ModelFormatInfo& info);
const char* model_format_name(ModelFormat format);

} // namespace pnnx

#endif // PNNX_MODEL_FORMAT_H
