// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>

#include <string>

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

static void write_current_pt2(const char* path)
{
    pnnx::StoreZipWriter writer;
    writer.open(path);
    write_record(writer, "model/archive_format", "pt2");
    write_record(writer, "model/archive_version", "1");
    write_record(writer, "model/models/model.json", "{}");
    writer.close();
}

static void write_unknown(const char* path)
{
    pnnx::StoreZipWriter writer;
    writer.open(path);
    write_record(writer, "model/file.txt", "unknown");
    writer.close();
}

int main()
{
    const char* torchscript_path = "test_model_format_torchscript.zip";
    const char* legacy_pt2_path = "test_model_format_legacy_pt2.zip";
    const char* current_pt2_path = "test_model_format_current_pt2.zip";
    const char* unknown_path = "test_model_format_unknown.zip";

    write_torchscript(torchscript_path);
    write_legacy_pt2(legacy_pt2_path);
    write_current_pt2(current_pt2_path);
    write_unknown(unknown_path);

    expect_format(torchscript_path, pnnx::ModelFormatTorchScript, "detect torchscript");
    expect_format(legacy_pt2_path, pnnx::ModelFormatExportedProgramLegacy, "detect legacy pt2");
    expect_format(current_pt2_path, pnnx::ModelFormatExportedProgram, "detect current pt2");
    expect_format(unknown_path, pnnx::ModelFormatUnknown, "reject unknown archive");

    remove(torchscript_path);
    remove(legacy_pt2_path);
    remove(current_pt2_path);
    remove(unknown_path);

    if (test_failures != 0)
    {
        fprintf(stderr, "%d model format test(s) failed\n", test_failures);
        return 1;
    }

    return 0;
}