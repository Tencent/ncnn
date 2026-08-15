// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "model_format.h"

#include <stdio.h>
#include <stdint.h>

#include "pt2_archive.h"

namespace pnnx {

int probe_model_format(const std::string& path, ModelFormatInfo& info)
{
    info.format = ModelFormatOther;
    info.archive_version.clear();
    info.diagnostic.clear();

    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        info.diagnostic = "cannot open model file";
        return -1;
    }

    uint32_t signature = 0;
    const size_t nread = fread(&signature, sizeof(signature), 1, fp);
    fclose(fp);

    if (nread != 1)
        return 0;

    if (signature != 0x04034b50 && signature != 0x06054b50 && signature != 0x06064b50)
        return 0;

    info.format = ModelFormatUnknownZip;

    Pt2ArchiveReader archive;
    if (archive.open(path) != 0)
    {
        info.diagnostic = archive.error;
        return -1;
    }

    info.archive_version = archive.archive_version;

    if (archive.container_kind == Pt2ContainerLegacyExportedProgram)
    {
        info.format = ModelFormatPt2LegacyExportedProgram;
        return 0;
    }
    if (archive.container_kind == Pt2ContainerArchive)
    {
        info.format = ModelFormatPt2Archive;
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
        info.format = ModelFormatTorchScript;
        return 0;
    }

    if (archive.has_compressed_records)
    {
        info.diagnostic = "unsupported compression method in ZIP container";
        return -1;
    }

    info.format = ModelFormatUnknownZip;
    info.diagnostic = "ZIP container is neither TorchScript nor a supported PT2 container";
    return 0;
}

const char* model_format_name(ModelFormat format)
{
    switch (format)
    {
    case ModelFormatOther:
        return "other";
    case ModelFormatTorchScript:
        return "torchscript";
    case ModelFormatPt2LegacyExportedProgram:
        return "pt2-legacy-exported-program";
    case ModelFormatPt2Archive:
        return "pt2-archive";
    case ModelFormatUnknownZip:
        return "unknown-zip";
    }

    return "unknown";
}

} // namespace pnnx
