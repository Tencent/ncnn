// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>

#include <string>
#include <vector>

#include "model_format.h"
#include "pt2_archive.h"
#include "storezip.h"

static bool parse_expected_format(const std::string& name, pnnx::ModelFormat& format)
{
    if (name == "other")
        format = pnnx::ModelFormatOther;
    else if (name == "torchscript")
        format = pnnx::ModelFormatTorchScript;
    else if (name == "pt2-legacy-exported-program")
        format = pnnx::ModelFormatPt2LegacyExportedProgram;
    else if (name == "pt2-archive")
        format = pnnx::ModelFormatPt2Archive;
    else if (name == "unknown-zip")
        format = pnnx::ModelFormatUnknownZip;
    else
        return false;
    return true;
}

int main(int argc, char** argv)
{
    if (argc == 3 && std::string(argv[1]) == "--roundtrip")
    {
        const char first[] = "first payload";
        const char second[] = "second payload";
        pnnx::StoreZipWriter writer;
        if (writer.open(argv[2]) != 0 ||
            writer.write_file("root/first", first, sizeof(first) - 1) != 0 ||
            writer.write_file("root/second", second, sizeof(second) - 1) != 0 ||
            writer.close() != 0)
        {
            fprintf(stderr, "StoreZipWriter roundtrip setup failed\n");
            return 1;
        }

        pnnx::StoreZipReader reader;
        std::vector<unsigned char> contents;
        if (reader.open(argv[2]) != 0 || reader.get_names().size() != 2 ||
            reader.read_file("root/second", contents) != 0 ||
            std::string(contents.begin(), contents.end()) != second)
        {
            fprintf(stderr, "StoreZipReader roundtrip failed: %s\n", reader.error.c_str());
            return 1;
        }
        return 0;
    }

    if (argc != 3)
    {
        fprintf(stderr, "usage: %s MODEL EXPECTED_FORMAT\n", argv[0]);
        return 2;
    }

    pnnx::ModelFormatInfo info;
    const int probe_result = pnnx::probe_model_format(argv[1], info);
    if (std::string(argv[2]) == "invalid-zip")
    {
        if (probe_result == 0)
        {
            fprintf(stderr, "expected invalid ZIP, got %s\n", pnnx::model_format_name(info.format));
            return 1;
        }
        return 0;
    }

    pnnx::ModelFormat expected_format;
    if (!parse_expected_format(argv[2], expected_format))
    {
        fprintf(stderr, "unknown expected format %s\n", argv[2]);
        return 2;
    }

    if (info.format != expected_format)
    {
        fprintf(stderr, "expected %s, got %s: %s\n", argv[2], pnnx::model_format_name(info.format), info.diagnostic.c_str());
        return 1;
    }
    if (probe_result != 0)
    {
        fprintf(stderr, "unexpected probe return %d for %s\n", probe_result, argv[2]);
        return 1;
    }

    if (expected_format == pnnx::ModelFormatPt2LegacyExportedProgram || expected_format == pnnx::ModelFormatPt2Archive)
    {
        pnnx::Pt2ArchiveReader archive;
        if (archive.open(argv[1]) != 0 || archive.model_record.empty())
        {
            fprintf(stderr, "PT2 archive open failed: %s\n", archive.error.c_str());
            return 1;
        }

        std::vector<unsigned char> model_json;
        if (archive.read_file(archive.model_record, model_json) != 0 || model_json.empty() || model_json[0] != '{')
        {
            fprintf(stderr, "PT2 model record read failed: %s\n", archive.error.c_str());
            return 1;
        }

        if (expected_format == pnnx::ModelFormatPt2Archive && archive.archive_version.empty())
        {
            fprintf(stderr, "PT2 archive version was not exposed\n");
            return 1;
        }
    }

    printf("%s archive_version=%s\n", pnnx::model_format_name(info.format), info.archive_version.c_str());
    if (!info.diagnostic.empty())
        printf("diagnostic=%s\n", info.diagnostic.c_str());
    return 0;
}
