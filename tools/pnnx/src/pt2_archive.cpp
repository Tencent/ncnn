// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_archive.h"

#include <algorithm>

namespace pnnx {

static std::string trim_ascii_whitespace(const std::string& value)
{
    size_t begin = 0;
    while (begin < value.size() && (value[begin] == ' ' || value[begin] == '\t' || value[begin] == '\r' || value[begin] == '\n'))
        begin++;

    size_t end = value.size();
    while (end > begin && (value[end - 1] == ' ' || value[end - 1] == '\t' || value[end - 1] == '\r' || value[end - 1] == '\n'))
        end--;

    return value.substr(begin, end - begin);
}

Pt2ArchiveReader::Pt2ArchiveReader()
{
    container_kind = Pt2ContainerUnknown;
}

int Pt2ArchiveReader::fail(const std::string& message)
{
    close();
    error = message;
    return -1;
}

int Pt2ArchiveReader::open(const std::string& path)
{
    close();

    if (zip.open(path) != 0)
        return fail(zip.error);

    const std::vector<std::string> physical_names = zip.get_names();

    // PyTorch's inline_container normally wraps all records in a basename
    // directory. Legacy ExportedProgram ZIPs are rootless. Expose one stable
    // logical namespace while retaining the physical name for record reads.
    if (!physical_names.empty())
    {
        const size_t slash = physical_names[0].find('/');
        if (slash != std::string::npos)
        {
            prefix = physical_names[0].substr(0, slash + 1);
            for (size_t i = 1; i < physical_names.size(); i++)
            {
                if (physical_names[i].compare(0, prefix.size(), prefix) != 0)
                {
                    prefix.clear();
                    break;
                }
            }
        }
    }

    for (size_t i = 0; i < physical_names.size(); i++)
    {
        std::string logical_name = physical_names[i];
        if (!prefix.empty())
            logical_name = physical_names[i].substr(prefix.size());

        if (logical_name.empty() || !records.insert(logical_name).second)
            return fail("duplicate or empty logical archive record: " + logical_name);
    }

    if (records.find("archive_format") != records.end())
    {
        std::string archive_format;
        if (read_small_text("archive_format", archive_format) != 0)
            return -1;

        if (trim_ascii_whitespace(archive_format) != "pt2")
            return 0;

        if (records.find("archive_version") == records.end())
            return fail("PT2 archive is missing archive_version");
        if (read_small_text("archive_version", archive_version) != 0)
            return -1;
        archive_version = trim_ascii_whitespace(archive_version);
        if (archive_version.empty())
            return fail("PT2 archive has an empty archive_version");

        model_record = "models/model.json";
        if (records.find(model_record) == records.end())
            return fail("PT2 archive is missing models/model.json");

        container_kind = Pt2ContainerArchive;
        return 0;
    }

    if (records.find("serialized_exported_program.json") != records.end())
    {
        if (records.find("version") == records.end())
            return fail("legacy PT2 archive is missing version");

        model_record = "serialized_exported_program.json";
        container_kind = Pt2ContainerLegacyExportedProgram;
    }

    return 0;
}

int Pt2ArchiveReader::close()
{
    zip.close();
    container_kind = Pt2ContainerUnknown;
    archive_version.clear();
    model_record.clear();
    records.clear();
    prefix.clear();
    error.clear();
    return 0;
}

int Pt2ArchiveReader::read_file(const std::string& logical_name, std::vector<unsigned char>& data)
{
    if (records.find(logical_name) == records.end())
        return fail("no such logical archive record: " + logical_name);
    if (zip.read_file(prefix + logical_name, data) != 0)
        return fail(zip.error);
    return 0;
}

int Pt2ArchiveReader::read_small_text(const std::string& logical_name, std::string& value)
{
    const std::string physical_name = prefix + logical_name;
    if (zip.get_file_size(physical_name) > 4096)
        return fail("archive metadata record is unexpectedly large: " + logical_name);

    std::vector<unsigned char> data;
    if (zip.read_file(physical_name, data) != 0)
        return fail(zip.error);
    if (std::find(data.begin(), data.end(), 0) != data.end())
        return fail("archive metadata record contains NUL: " + logical_name);

    value.assign(data.begin(), data.end());
    return 0;
}

} // namespace pnnx
