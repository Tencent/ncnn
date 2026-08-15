// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PT2_ARCHIVE_H
#define PNNX_PT2_ARCHIVE_H

#include <set>
#include <string>
#include <vector>

#include "storezip.h"

namespace pnnx {

enum Pt2ContainerKind
{
    Pt2ContainerUnknown,
    Pt2ContainerLegacyExportedProgram,
    Pt2ContainerArchive
};

class Pt2ArchiveReader
{
public:
    Pt2ArchiveReader();

    // Opens a stored ZIP archive and probes its logical records. A successful
    // open may still have Pt2ContainerUnknown, allowing the model-format probe
    // to distinguish TorchScript and unrelated ZIP files without reopening it.
    int open(const std::string& path);
    int close();

    int read_file(const std::string& logical_name, std::vector<unsigned char>& data);

    Pt2ContainerKind container_kind;
    std::string archive_version;
    std::string model_record;
    std::set<std::string> records;
    bool has_compressed_records;
    std::string error;

private:
    int fail(const std::string& message);
    int read_small_text(const std::string& logical_name, std::string& text);

    StoreZipReader zip;
    std::string prefix;
};

} // namespace pnnx

#endif // PNNX_PT2_ARCHIVE_H
