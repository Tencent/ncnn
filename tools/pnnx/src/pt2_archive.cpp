// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_archive.h"

#include "utils.h"

namespace pnnx {

Pt2ArchiveReader::Pt2ArchiveReader()
{
    container_kind = Pt2ContainerUnknown;
    has_compressed_records = false;
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
    has_compressed_records = zip.has_compressed_records;

    const std::vector<std::string> physical_names = zip.get_names();

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

    if (prefix.empty())
    {
        records.insert(physical_names.begin(), physical_names.end());
    }
    else
    {
        for (size_t i = 0; i < physical_names.size(); i++)
            records.insert(physical_names[i].substr(prefix.size()));
    }

    std::vector<unsigned char> data;
    if (records.find("archive_format") != records.end())
    {
        if (read_file("archive_format", data) != 0)
            return -1;
        if (trim_ascii_whitespace(std::string(data.begin(), data.end())) != "pt2")
            return 0;

        if (records.find("archive_version") == records.end())
            return fail("PT2 archive is missing archive_version");
        if (read_file("archive_version", data) != 0)
            return -1;
        archive_version.assign(data.begin(), data.end());
        archive_version = trim_ascii_whitespace(archive_version);
        if (archive_version.empty())
            return fail("PT2 archive has an empty archive_version");
        if (archive_version != "0")
            return fail("unsupported PT2 archive version: " + archive_version);

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
        if (read_file("version", data) != 0)
            return -1;
        archive_version.assign(data.begin(), data.end());
        archive_version = trim_ascii_whitespace(archive_version);
        if (archive_version.empty())
            return fail("legacy PT2 archive has an empty version");

        model_record = "serialized_exported_program.json";
        container_kind = Pt2ContainerLegacyExportedProgram;
    }

    return 0;
}

void Pt2ArchiveReader::close()
{
    zip.close();
    container_kind = Pt2ContainerUnknown;
    archive_version.clear();
    model_record.clear();
    records.clear();
    prefix.clear();
    has_compressed_records = false;
    error.clear();
}

int Pt2ArchiveReader::read_file(const std::string& logical_name, std::vector<unsigned char>& data)
{
    if (records.find(logical_name) == records.end())
        return fail("no such logical archive record: " + logical_name);
    if (zip.read_file(prefix + logical_name, data) != 0)
        return fail(zip.error);
    return 0;
}

} // namespace pnnx
