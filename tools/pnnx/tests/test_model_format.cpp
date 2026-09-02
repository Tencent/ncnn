// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>

#include <fstream>
#include <string>
#include <vector>

#include "model_format.h"
#include "storezip.h"

static int test_failures = 0;

static void expect_format(const char* path, pnnx::ModelFormat expected, const char* message)
{
    std::string error;
    const pnnx::ModelFormat actual = pnnx::detect_model_format(path, error);
    if (actual == expected)
        return;

    fprintf(stderr, "FAILED: %s: %s\n", message, error.c_str());
    test_failures++;
}

static void expect_unknown(const char* path, const char* message)
{
    std::string error;
    const pnnx::ModelFormat actual = pnnx::detect_model_format(path, error);
    if (actual == pnnx::ModelFormatUnknown && !error.empty())
        return;

    fprintf(stderr, "FAILED: %s: format=%d error=%s\n", message, (int)actual, error.c_str());
    test_failures++;
}

static void write_record(pnnx::StoreZipWriter& writer, const char* name, const char* value)
{
    writer.write_file(name, value, std::string(value).size());
}

static void write_torchscript(const char* path)
{
    pnnx::StoreZipWriter writer;
    writer.open(path);
    write_record(writer, "model/data.pkl", "data");
    write_record(writer, "model/constants.pkl", "constants");
    writer.close();
}

static void write_legacy_pt2(const char* path)
{
    pnnx::StoreZipWriter writer;
    writer.open(path);
    write_record(writer, "serialized_exported_program.json", "{}");
    write_record(writer, "serialized_state_dict.pt", "state");
    write_record(writer, "serialized_constants.pt", "constants");
    write_record(writer, "serialized_example_inputs.pt", "inputs");
    write_record(writer, "version", "8.2");
    writer.close();
}

static void write_current_pt2(const char* path, const char* format = "pt2", const char* version = "0", bool include_model = true, bool include_unknown = false)
{
    pnnx::StoreZipWriter writer;
    writer.open(path);
    write_record(writer, "model/archive_format", format);
    write_record(writer, "model/archive_version", version);
    if (include_model)
        write_record(writer, "model/models/model.json", "{}");
    if (include_unknown)
        write_record(writer, "model/extra/future_field", "ignored");
    writer.close();
}

static void write_unknown(const char* path)
{
    pnnx::StoreZipWriter writer;
    writer.open(path);
    write_record(writer, "model/file.txt", "unknown");
    writer.close();
}

static void write_truncated(const char* path)
{
    const unsigned char data[] = {0x50, 0x4b, 0x03, 0x04};
    std::ofstream output(path, std::ios::binary);
    output.write((const char*)data, sizeof(data));
}

static void overwrite_u16(const char* path, long offset, uint16_t value)
{
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(offset);
    file.write((const char*)&value, sizeof(value));
}

static void truncate_tail(const char* path, size_t size)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const std::streamoff file_size = input.tellg();
    input.seekg(0);

    std::vector<char> data((size_t)file_size - size);
    input.read(data.data(), data.size());
    input.close();

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(data.data(), data.size());
}

int main()
{
    const char* torchscript_path = "test_model_format_torchscript.zip";
    const char* legacy_pt2_path = "test_model_format_legacy_pt2.zip";
    const char* current_pt2_path = "test_model_format_current_pt2.zip";
    const char* current_pt2_unknown_field_path = "test_model_format_current_pt2_unknown_field.zip";
    const char* wrong_format_path = "test_model_format_wrong_format.zip";
    const char* wrong_version_path = "test_model_format_wrong_version.zip";
    const char* incomplete_pt2_path = "test_model_format_incomplete_pt2.zip";
    const char* truncated_path = "test_model_format_truncated.zip";
    const char* missing_eocd_path = "test_model_format_missing_eocd.zip";
    const char* data_descriptor_path = "test_model_format_data_descriptor.zip";
    const char* compressed_path = "test_model_format_compressed.zip";
    const char* unknown_path = "test_model_format_unknown.zip";

    write_torchscript(torchscript_path);
    write_legacy_pt2(legacy_pt2_path);
    write_current_pt2(current_pt2_path);
    write_current_pt2(current_pt2_unknown_field_path, "pt2", "0", true, true);
    write_current_pt2(wrong_format_path, "future");
    write_current_pt2(wrong_version_path, "pt2", "1");
    write_current_pt2(incomplete_pt2_path, "pt2", "0", false);
    write_truncated(truncated_path);
    write_current_pt2(missing_eocd_path);
    truncate_tail(missing_eocd_path, 22);
    write_current_pt2(data_descriptor_path);
    overwrite_u16(data_descriptor_path, 6, 8);
    write_current_pt2(compressed_path);
    overwrite_u16(compressed_path, 8, 8);
    write_unknown(unknown_path);

    expect_format(torchscript_path, pnnx::ModelFormatTorchScript, "detect torchscript");
    expect_format(legacy_pt2_path, pnnx::ModelFormatExportedProgramLegacy, "detect legacy pt2");
    expect_format(current_pt2_path, pnnx::ModelFormatExportedProgram, "detect current pt2");
    expect_format(current_pt2_unknown_field_path, pnnx::ModelFormatExportedProgram, "ignore unknown pt2 field");
    expect_unknown(wrong_format_path, "reject unsupported archive format");
    expect_unknown(wrong_version_path, "reject unsupported archive version");
    expect_unknown(incomplete_pt2_path, "reject incomplete pt2 archive");
    expect_unknown(truncated_path, "reject truncated archive");
    expect_unknown(missing_eocd_path, "reject archive without end of central directory");
    expect_unknown(data_descriptor_path, "reject data descriptor");
    expect_unknown(compressed_path, "reject compressed archive");
    expect_unknown(unknown_path, "reject unknown archive");

    remove(torchscript_path);
    remove(legacy_pt2_path);
    remove(current_pt2_path);
    remove(current_pt2_unknown_field_path);
    remove(wrong_format_path);
    remove(wrong_version_path);
    remove(incomplete_pt2_path);
    remove(truncated_path);
    remove(missing_eocd_path);
    remove(data_descriptor_path);
    remove(compressed_path);
    remove(unknown_path);

    if (test_failures != 0)
    {
        fprintf(stderr, "%d model format test(s) failed\n", test_failures);
        return 1;
    }

    return 0;
}