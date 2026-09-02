// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "model_format.h"

#include <map>
#include <vector>

#include "storezip.h"

namespace pnnx {

static std::string get_common_archive_root(const std::vector<std::string>& names)
{
    if (names.empty())
        return std::string();

    const size_t slash = names[0].find('/');
    if (slash == std::string::npos)
        return std::string();

    const std::string root = names[0].substr(0, slash + 1);
    for (size_t i = 1; i < names.size(); i++)
    {
        if (names[i].compare(0, root.size(), root) != 0)
            return std::string();
    }

    return root;
}

static std::map<std::string, std::string> get_logical_records(const std::vector<std::string>& names)
{
    const std::string root = get_common_archive_root(names);

    std::map<std::string, std::string> records;
    for (size_t i = 0; i < names.size(); i++)
    {
        const std::string logical_name = root.empty() ? names[i] : names[i].substr(root.size());
        records[logical_name] = names[i];
    }

    return records;
}

static bool has_record(const std::map<std::string, std::string>& records, const char* name)
{
    return records.find(name) != records.end();
}

static bool has_exported_program(const std::map<std::string, std::string>& records)
{
    for (std::map<std::string, std::string>::const_iterator it = records.begin(); it != records.end(); ++it)
    {
        const std::string& name = it->first;
        if (name.size() > 12 && name.compare(0, 7, "models/") == 0 && name.compare(name.size() - 5, 5, ".json") == 0)
            return true;
    }

    return false;
}

static bool read_small_record(StoreZipReader& reader, const std::string& name, std::string& value)
{
    const uint64_t size = reader.get_file_size(name);
    if (size == 0 || size > 64)
        return false;

    value.resize((size_t)size);
    return reader.read_file(name, &value[0]) == 0;
}

ModelFormat detect_model_format(const std::string& path, std::string& error)
{
    error.clear();

    StoreZipReader reader;
    if (reader.open(path) != 0)
    {
        error = "failed to read model zip archive";
        return ModelFormatUnknown;
    }

    const std::map<std::string, std::string> records = get_logical_records(reader.get_names());

    std::map<std::string, std::string>::const_iterator archive_format = records.find("archive_format");
    if (archive_format != records.end())
    {
        std::string value;
        if (!read_small_record(reader, archive_format->second, value))
        {
            error = "failed to read pt2 archive format";
            return ModelFormatUnknown;
        }

        if (value != "pt2")
        {
            error = "unsupported model archive format " + value;
            return ModelFormatUnknown;
        }

        if (!has_record(records, "archive_version") || !has_exported_program(records))
        {
            error = "incomplete pt2 model archive";
            return ModelFormatUnknown;
        }

        std::string version;
        if (!read_small_record(reader, records.find("archive_version")->second, version))
        {
            error = "failed to read pt2 archive version";
            return ModelFormatUnknown;
        }

        if (version != "0")
        {
            error = "unsupported pt2 archive version " + version;
            return ModelFormatUnknown;
        }

        return ModelFormatExportedProgram;
    }

    if (has_record(records, "serialized_exported_program.json")
        && has_record(records, "serialized_state_dict.pt")
        && has_record(records, "serialized_constants.pt")
        && has_record(records, "serialized_example_inputs.pt")
        && has_record(records, "version"))
    {
        return ModelFormatExportedProgramLegacy;
    }

    if (has_record(records, "data.pkl") && has_record(records, "constants.pkl"))
        return ModelFormatTorchScript;

    error = "unsupported model zip archive";
    return ModelFormatUnknown;
}

} // namespace pnnx