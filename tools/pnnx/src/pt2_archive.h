// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PT2_ARCHIVE_H
#define PNNX_PT2_ARCHIVE_H

#include "json_reader.h"
#include "storezip.h"

#include <stdint.h>

#include <string>
#include <vector>

namespace pnnx {

struct ModelFormatInfo;

enum Pt2ByteOrder
{
    PT2_BYTE_ORDER_LITTLE,
    PT2_BYTE_ORDER_BIG
};

struct Pt2PackageLayout
{
    Pt2PackageLayout();

    std::string root;
    uint64_t archive_version;
    std::string model_name;
    std::string model_json_path;
    std::string weights_config_path;
    std::string constants_config_path;
    Pt2ByteOrder byte_order;
};

class Pt2ArchiveReader
{
public:
    Pt2ArchiveReader();

    int open(const std::string& path, const ModelFormatInfo& format_info, std::string& error);

    const Pt2PackageLayout& layout() const;

    int read_json(const std::string& entry, JsonValue& value, std::string& error);
    int read_blob(const std::string& entry, std::vector<char>& data, std::string& error);

private:
    Pt2ArchiveReader(const Pt2ArchiveReader&) = delete;
    Pt2ArchiveReader& operator=(const Pt2ArchiveReader&) = delete;

    void reset();

    StoreZipReader reader;
    Pt2PackageLayout package_layout;
    bool opened;
};

} // namespace pnnx

#endif // PNNX_PT2_ARCHIVE_H
