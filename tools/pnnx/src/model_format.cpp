// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "model_format.h"

#include "storezip.h"

#include <stdio.h>
#include <string.h>

#include <limits>
#include <string>
#include <vector>

namespace pnnx {

static int model_file_is_zip_candidate(const std::string& path, bool& is_zip, std::string& error)
{
    is_zip = false;

    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        error = "open model file failed " + path;
        return -1;
    }

    unsigned char signature[4];
    const size_t read_count = fread(signature, 1, sizeof(signature), fp);
    const bool read_failed = ferror(fp) != 0;
    fclose(fp);

    if (read_failed)
    {
        error = "read model file signature failed " + path;
        return -1;
    }

    if (read_count != sizeof(signature))
        return 0;

    const bool local_file_header = signature[0] == 'P' && signature[1] == 'K' && signature[2] == 0x03 && signature[3] == 0x04;
    const bool empty_archive = signature[0] == 'P' && signature[1] == 'K' && signature[2] == 0x05 && signature[3] == 0x06;
    const bool zip64_archive = signature[0] == 'P' && signature[1] == 'K' && signature[2] == 0x06 && signature[3] == 0x06;
    is_zip = local_file_header || empty_archive || zip64_archive;

    return 0;
}

static bool has_suffix(const std::string& value, const std::string& suffix)
{
    if (value.size() <= suffix.size())
        return false;

    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int detect_model_format(const std::string& path, ModelFormatInfo& info, std::string& error)
{
    info.format = MODEL_FORMAT_UNKNOWN;
    info.archive_root.clear();
    info.archive_version = 0;
    error.clear();

    bool is_zip = false;
    if (model_file_is_zip_candidate(path, is_zip, error) != 0)
        return -1;

    if (!is_zip)
        return 0;

    StoreZipReader reader;
    if (reader.open(path) != 0)
    {
        error = "invalid zip model archive " + path;
        return -1;
    }

    const std::string archive_format_suffix = "/archive_format";
    std::string archive_format_name;

    std::vector<std::string> names;
    if (reader.get_names(names) != 0)
    {
        error = "cannot enumerate zip model archive entries";
        return -1;
    }
    for (size_t i = 0; i < names.size(); i++)
    {
        const std::string& name = names[i];
        if (!has_suffix(name, archive_format_suffix))
            continue;

        if (!archive_format_name.empty())
        {
            error = "multiple archive_format entries in zip model archive";
            return -1;
        }

        archive_format_name = name;
    }

    if (archive_format_name.empty())
    {
        info.format = MODEL_FORMAT_TORCHSCRIPT;
        return 0;
    }

    if (!reader.is_file_stored(archive_format_name))
    {
        error = archive_format_name + " must use zip store compression";
        return -1;
    }

    if (reader.get_file_size(archive_format_name) != 3)
    {
        error = "archive_format payload is not pt2";
        return -1;
    }

    char archive_format[3];
    if (reader.read_file(archive_format_name, archive_format) != 0)
    {
        error = "cannot read " + archive_format_name;
        return -1;
    }

    if (memcmp(archive_format, "pt2", 3) != 0)
    {
        error = "archive_format payload is not pt2";
        return -1;
    }

    const std::string archive_root = archive_format_name.substr(0, archive_format_name.size() - archive_format_suffix.size());
    const std::string archive_version_name = archive_root + "/archive_version";
    if (!reader.has_file(archive_version_name))
    {
        error = archive_version_name + " is missing";
        return -1;
    }
    if (!reader.is_file_stored(archive_version_name))
    {
        error = archive_version_name + " must use zip store compression";
        return -1;
    }

    const uint64_t archive_version_size = reader.get_file_size(archive_version_name);
    if (archive_version_size == 0)
    {
        error = "archive_version is not an unsigned decimal integer";
        return -1;
    }
    if (archive_version_size > 20)
    {
        error = "archive_version payload is longer than 20 bytes";
        return -1;
    }

    std::vector<char> archive_version_text((size_t)archive_version_size);
    if (reader.read_file(archive_version_name, archive_version_text.data()) != 0)
    {
        error = "cannot read " + archive_version_name;
        return -1;
    }

    uint64_t archive_version = 0;
    for (size_t i = 0; i < archive_version_text.size(); i++)
    {
        const unsigned char ch = (unsigned char)archive_version_text[i];
        if (ch < '0' || ch > '9')
        {
            error = "archive_version is not an unsigned decimal integer";
            return -1;
        }

        const uint64_t digit = ch - '0';
        if (archive_version > (std::numeric_limits<uint64_t>::max() - digit) / 10)
        {
            error = "archive_version is out of uint64 range";
            return -1;
        }

        archive_version = archive_version * 10 + digit;
    }

    info.format = MODEL_FORMAT_EXPORTED_PROGRAM_PT2;
    info.archive_root = archive_root;
    info.archive_version = archive_version;

    return 0;
}

} // namespace pnnx
