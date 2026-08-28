// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "model_format.h"

#include <stdio.h>
#include <stdint.h>

#include "pt2_archive.h"

namespace pnnx {

int probe_model_format(const std::string& path, ModelFormatInfo& info)
{
    info.format = Other;
    info.diagnostic.clear();

    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        info.diagnostic = "cannot open model file";
        return -1;
    }

    uint32_t signature = 0;
    fread((char*)&signature, sizeof(signature), 1, fp);
    fclose(fp);

    if (signature != 0x04034b50 && signature != 0x06054b50 && signature != 0x06064b50)
        return 0;

    info.format = UnknownZip;

    Pt2ArchiveReader archive;
    if (archive.open(path) != 0)
    {
        info.diagnostic = archive.error;
        return -1;
    }

    if (archive.container_kind != Pt2ContainerUnknown)
    {
        info.format = Pt2;
        return 0;
    }

    bool has_code = false;
    for (std::set<std::string>::const_iterator it = archive.records.begin(); it != archive.records.end(); ++it)
    {
        if (it->compare(0, 5, "code/") == 0)
        {
            has_code = true;
            break;
        }
    }

    if (archive.records.find("data.pkl") != archive.records.end() &&
        (archive.records.find("version") != archive.records.end() || archive.records.find(".data/version") != archive.records.end()) &&
        (has_code || archive.records.find("constants.pkl") != archive.records.end()))
    {
        info.format = TorchScript;
        return 0;
    }

    if (archive.has_compressed_records)
    {
        info.diagnostic = "unsupported compression method in ZIP container";
        return -1;
    }

    info.diagnostic = "ZIP container is neither TorchScript nor a supported PT2 container";
    return 0;
}

const char* model_format_name(ModelFormat format)
{
    switch (format)
    {
    case Other:
        return "other";
    case TorchScript:
        return "torchscript";
    case Pt2:
        return "pt2";
    case UnknownZip:
        return "unknown-zip";
    }

    return "unknown";
}

} // namespace pnnx
