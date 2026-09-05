// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_MODEL_FORMAT_H
#define PNNX_MODEL_FORMAT_H

#include <stdint.h>
#include <string>

namespace pnnx {

enum ModelFormat
{
    MODEL_FORMAT_UNKNOWN,
    MODEL_FORMAT_TORCHSCRIPT,
    MODEL_FORMAT_EXPORTED_PROGRAM_PT2
};

struct ModelFormatInfo
{
    ModelFormat format;
    std::string archive_root;
    uint64_t archive_version;
};

int detect_model_format(const std::string& path, ModelFormatInfo& info, std::string& error);

} // namespace pnnx

#endif // PNNX_MODEL_FORMAT_H
