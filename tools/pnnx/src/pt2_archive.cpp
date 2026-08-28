// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_archive.h"

#include "model_format.h"

#include <stddef.h>

#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pnnx {

static bool starts_with(const std::string& value, const std::string& prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

static bool ends_with(const std::string& value, const std::string& suffix)
{
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static int read_stored_entry(StoreZipReader& reader, const std::string& entry, uint64_t size_limit, std::vector<char>& data, std::string& error)
{
    data.clear();

    if (!reader.has_file(entry))
    {
        error = entry + " is missing";
        return -1;
    }

    if (!reader.is_file_stored(entry))
    {
        error = "compressed pt2 entry is unsupported " + entry;
        return -1;
    }

    const uint64_t entry_size = reader.get_file_size(entry);
    if (entry_size > size_limit)
    {
        std::ostringstream message;
        message << entry << " exceeds the " << size_limit << " byte read limit";
        error = message.str();
        return -1;
    }

    if (entry_size > (uint64_t)std::numeric_limits<size_t>::max())
    {
        error = entry + " is too large for this platform";
        return -1;
    }

    try
    {
        data.resize((size_t)entry_size);
    }
    catch (const std::length_error&)
    {
        error = entry + " is too large for this platform";
        return -1;
    }
    catch (const std::bad_alloc&)
    {
        error = "cannot allocate memory for " + entry;
        return -1;
    }

    char* data_pointer = data.empty() ? 0 : &data[0];
    if (reader.read_file(entry, data_pointer) != 0)
    {
        data.clear();
        error = "cannot read " + entry;
        return -1;
    }

    return 0;
}

Pt2PackageLayout::Pt2PackageLayout()
{
    archive_version = 0;
    byte_order = PT2_BYTE_ORDER_LITTLE;
}

Pt2ArchiveReader::Pt2ArchiveReader()
{
    opened = false;
}

void Pt2ArchiveReader::reset()
{
    reader.close();

    package_layout.root.clear();
    package_layout.archive_version = 0;
    package_layout.model_name.clear();
    package_layout.model_json_path.clear();
    package_layout.weights_config_path.clear();
    package_layout.constants_config_path.clear();
    package_layout.byte_order = PT2_BYTE_ORDER_LITTLE;
    opened = false;
}

int Pt2ArchiveReader::open(const std::string& path, const ModelFormatInfo& format_info, std::string& error)
{
    reset();
    error.clear();

    if (format_info.format != MODEL_FORMAT_EXPORTED_PROGRAM_PT2)
    {
        error = "model archive is not an exported program pt2 package";
        return -1;
    }

    if (format_info.archive_version != 0)
    {
        std::ostringstream message;
        message << "unsupported pt2 archive version " << format_info.archive_version;
        error = message.str();
        return -1;
    }

    if (reader.open(path) != 0)
    {
        error = "cannot reopen exported program pt2 package " + path;
        reset();
        return -1;
    }

    const std::string root_prefix = format_info.archive_root + "/";
    const std::string models_prefix = root_prefix + "models/";
    const std::string aotinductor_prefix = root_prefix + "data/aotinductor/";

    std::vector<std::string> model_entries;
    bool has_aotinductor_entry = false;

    std::vector<std::string> names;
    if (reader.get_names(names) != 0)
    {
        error = "cannot enumerate exported program archive entries";
        reset();
        return -1;
    }
    for (size_t i = 0; i < names.size(); i++)
    {
        const std::string& name = names[i];
        if (!starts_with(name, root_prefix))
            continue;

        if (!reader.is_file_stored(name))
        {
            error = "compressed pt2 entry is unsupported " + name;
            reset();
            return -1;
        }

        if (starts_with(name, aotinductor_prefix))
            has_aotinductor_entry = true;

        if (!starts_with(name, models_prefix) || name == models_prefix)
            continue;

        const std::string relative_name = name.substr(models_prefix.size());
        if (relative_name.find('/') != std::string::npos || relative_name.find('\\') != std::string::npos || relative_name.size() <= 5 || !ends_with(relative_name, ".json"))
        {
            error = "invalid ExportedProgram model entry " + name;
            reset();
            return -1;
        }

        model_entries.push_back(name);
    }

    if (model_entries.empty())
    {
        error = has_aotinductor_entry ? "AOTInductor-only pt2 package is unsupported" : "pt2 package contains no ExportedProgram model";
        reset();
        return -1;
    }

    if (model_entries.size() != 1)
    {
        error = "multiple ExportedPrograms in one pt2 package are unsupported";
        reset();
        return -1;
    }

    const std::string& model_json_path = model_entries[0];
    const std::string model_filename = model_json_path.substr(models_prefix.size());
    const std::string model_name = model_filename.substr(0, model_filename.size() - 5);
    const std::string weights_config_path = root_prefix + "data/weights/" + model_name + "_weights_config.json";
    const std::string constants_config_path = root_prefix + "data/constants/" + model_name + "_constants_config.json";
    const std::string legacy_weights_path = root_prefix + "data/weights/" + model_name + ".pt";
    const std::string legacy_constants_path = root_prefix + "data/constants/" + model_name + ".pt";
    const std::string byteorder_path = root_prefix + "byteorder";

    if (!reader.has_file(weights_config_path) && !reader.has_file(constants_config_path) && reader.has_file(legacy_weights_path) && reader.has_file(legacy_constants_path))
    {
        error = "PyTorch 2.8 legacy pickled-payload PT2 is unsupported";
        reset();
        return -1;
    }

    if (!reader.has_file(weights_config_path))
    {
        error = weights_config_path + " is missing";
        reset();
        return -1;
    }
    if (!reader.has_file(constants_config_path))
    {
        error = constants_config_path + " is missing";
        reset();
        return -1;
    }

    std::vector<char> byteorder_data;
    if (read_stored_entry(reader, byteorder_path, 6, byteorder_data, error) != 0)
    {
        reset();
        return -1;
    }

    Pt2ByteOrder byte_order;
    const std::string byteorder_text(byteorder_data.begin(), byteorder_data.end());
    if (byteorder_text == "little")
    {
        byte_order = PT2_BYTE_ORDER_LITTLE;
    }
    else if (byteorder_text == "big")
    {
        byte_order = PT2_BYTE_ORDER_BIG;
    }
    else
    {
        error = "unsupported pt2 byteorder " + byteorder_text;
        reset();
        return -1;
    }

    package_layout.root = format_info.archive_root;
    package_layout.archive_version = format_info.archive_version;
    package_layout.model_name = model_name;
    package_layout.model_json_path = model_json_path;
    package_layout.weights_config_path = weights_config_path;
    package_layout.constants_config_path = constants_config_path;
    package_layout.byte_order = byte_order;
    opened = true;

    return 0;
}

const Pt2PackageLayout& Pt2ArchiveReader::layout() const
{
    return package_layout;
}

int Pt2ArchiveReader::read_json(const std::string& entry, JsonValue& value, std::string& error)
{
    error.clear();

    if (!opened)
    {
        error = "pt2 package is not open";
        return -1;
    }

    if (!reader.has_file(entry))
    {
        error = entry + " is missing";
        return -1;
    }

    const uint64_t max_json_size = (uint64_t)512 * 1024 * 1024;
    if (reader.get_file_size(entry) > max_json_size)
    {
        std::ostringstream message;
        message << entry << " exceeds the " << max_json_size << " byte json limit";
        error = message.str();
        return -1;
    }

    std::vector<char> data;
    if (read_blob(entry, data, error) != 0)
        return -1;

    JsonParseError parse_error;
    JsonParseOptions options;
    const char* data_pointer = data.empty() ? 0 : &data[0];
    if (parse_json(data_pointer, data.size(), value, parse_error, options) != 0)
    {
        std::ostringstream message;
        message << "invalid json " << entry
                << " at line " << parse_error.line
                << " column " << parse_error.column
                << " byte " << parse_error.byte_offset
                << ": " << parse_error.message;
        error = message.str();
        return -1;
    }

    return 0;
}

int Pt2ArchiveReader::read_blob(const std::string& entry, std::vector<char>& data, std::string& error)
{
    error.clear();
    data.clear();

    if (!opened)
    {
        error = "pt2 package is not open";
        return -1;
    }

    const std::string root_prefix = package_layout.root + "/";
    if (!starts_with(entry, root_prefix))
    {
        error = "pt2 entry is outside archive root " + entry;
        return -1;
    }

    return read_stored_entry(reader, entry, std::numeric_limits<uint64_t>::max(), data, error);
}

} // namespace pnnx
