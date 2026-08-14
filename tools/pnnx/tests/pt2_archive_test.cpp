// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <string.h>

#include <map>
#include <string>
#include <vector>

#include "model_format.h"
#include "pt2_archive.h"
#include "pt2_json.h"
#include "pt2_program.h"
#include "pt2_weights.h"
#include "storezip.h"

static int read_file(const char* path, std::vector<unsigned char>& data)
{
    FILE* fp = fopen(path, "rb");
    if (!fp)
        return -1;

    unsigned char buffer[4096];
    size_t nread;
    while ((nread = fread(buffer, 1, sizeof(buffer), fp)) != 0)
        data.insert(data.end(), buffer, buffer + nread);
    const int result = ferror(fp) ? -1 : 0;
    fclose(fp);
    return result;
}

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

static bool check_f32(const pnnx::Pt2Weights& weights, const char* name, pnnx::Pt2InputSpec::Kind kind,
                      const std::vector<int>& shape, const std::vector<float>& expected)
{
    std::map<std::string, pnnx::Pt2Weight>::const_iterator it = weights.values.find(name);
    if (it == weights.values.end() || it->second.kind != kind || it->second.attribute.type != 1 ||
        it->second.attribute.shape != shape || it->second.attribute.data.size() != expected.size() * sizeof(float))
        return false;
    for (size_t i = 0; i < expected.size(); i++)
    {
        float value;
        memcpy(&value, &it->second.attribute.data[i * sizeof(float)], sizeof(float));
        if (value != expected[i])
            return false;
    }
    return true;
}

static int check_weights(const pnnx::Pt2Weights& weights, const std::string& name)
{
    if (name == "structured_io")
        return weights.values.empty() ? 0 : -1;

    if (name == "state_and_constants")
    {
        std::vector<float> weight(12);
        for (size_t i = 0; i < weight.size(); i++)
            weight[i] = i / 16.f;
        return weights.values.size() == 5 &&
                       check_f32(weights, "weight", pnnx::Pt2InputSpec::Parameter, std::vector<int>{4, 3}, weight) &&
                       check_f32(weights, "bias", pnnx::Pt2InputSpec::Parameter, std::vector<int>{4}, std::vector<float>{0.f, .125f, .25f, .375f}) &&
                       check_f32(weights, "persistent_buffer", pnnx::Pt2InputSpec::Buffer, std::vector<int>{4}, std::vector<float>{.25f, -.5f, .75f, -1.f}) &&
                       check_f32(weights, "non_persistent_buffer", pnnx::Pt2InputSpec::Buffer, std::vector<int>{4}, std::vector<float>{1.f, 1.5f, 2.f, 2.5f}) &&
                       check_f32(weights, "tensor_constant", pnnx::Pt2InputSpec::TensorConstant, std::vector<int>{4}, std::vector<float>{.125f, .25f, .375f, .5f})
                   ? 0
                   : -1;
    }

    if (name == "strided_tensors")
    {
        std::vector<float> weight(30);
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 5; j++)
                weight[i * 5 + j] = j * 6.f + i;
        }
        return weights.values.size() == 3 &&
                       check_f32(weights, "weight", pnnx::Pt2InputSpec::Parameter, std::vector<int>{6, 5}, weight) &&
                       check_f32(weights, "offset_view", pnnx::Pt2InputSpec::Buffer, std::vector<int>{5}, std::vector<float>{3.f, 4.f, 5.f, 6.f, 7.f}) &&
                       check_f32(weights, "strided_view", pnnx::Pt2InputSpec::Buffer, std::vector<int>{5}, std::vector<float>{3.f, 5.f, 7.f, 9.f, 11.f})
                   ? 0
                   : -1;
    }

    if (name == "scalar_types")
    {
        const int attribute_types[] = {0, 8, 7, 6, 4, 5, 3, 1, 2, 12, 10, 11, 9, 13};
        const int element_sizes[] = {0, 1, 1, 2, 4, 8, 2, 4, 8, 4, 8, 16, 1, 2};
        if (weights.values.size() != 15)
            return -1;
        for (int dtype = 1; dtype <= 13; dtype++)
        {
            const pnnx::Attribute& attribute = weights.values.at("dtype_" + std::to_string(dtype)).attribute;
            if (attribute.type != attribute_types[dtype] || attribute.shape != std::vector<int>{2} || attribute.data.size() != (size_t)element_sizes[dtype] * 2)
                return -1;
            for (size_t i = 0; i < attribute.data.size(); i++)
            {
                if ((unsigned char)attribute.data[i] != i)
                    return -1;
            }
        }
        return weights.values.at("scalar").attribute.shape.empty() && weights.values.at("scalar").attribute.data.size() == 4 &&
                       weights.values.at("empty").attribute.shape == std::vector<int>{0} && weights.values.at("empty").attribute.data.empty()
                   ? 0
                   : -1;
    }

    if (name == "big_endian")
    {
        if (weights.values.size() != 1)
            return -1;
        float value;
        const pnnx::Attribute& attribute = weights.values.begin()->second.attribute;
        if (attribute.data.size() != sizeof(float))
            return -1;
        memcpy(&value, &attribute.data[0], sizeof(float));
        return value == 1.f ? 0 : -1;
    }
    return -1;
}

int main(int argc, char** argv)
{
    if (argc == 4 && std::string(argv[1]) == "--weights-archive")
    {
        pnnx::Pt2ArchiveReader archive;
        pnnx::Pt2Program program;
        pnnx::Pt2Weights weights;
        if (archive.open(argv[2]) != 0 || pnnx::load_pt2_program(archive, program) != 0 || pnnx::load_pt2_weights(archive, program, weights) != 0)
        {
            fprintf(stderr, "%s\n", !weights.error.empty() ? weights.error.c_str() : !program.error.empty() ? program.error.c_str() : archive.error.c_str());
            return 1;
        }
        if (check_weights(weights, argv[3]) != 0)
        {
            fprintf(stderr, "PT2 weights differ from expected values\n");
            return 1;
        }
        return 0;
    }

    if (argc == 3 && std::string(argv[1]) == "--program-archive")
    {
        pnnx::Pt2ArchiveReader archive;
        pnnx::Pt2Program program;
        if (archive.open(argv[2]) != 0 || pnnx::load_pt2_program(archive, program) != 0)
        {
            fprintf(stderr, "%s\n", archive.error.empty() ? program.error.c_str() : archive.error.c_str());
            return 1;
        }
        if (program.schema_major != 8 || program.opset_versions["aten"] != 10 || program.nodes.empty() || program.tensors.empty())
        {
            fprintf(stderr, "PT2 program decode returned incomplete data\n");
            return 1;
        }
        return 0;
    }

    if (argc == 3 && (std::string(argv[1]) == "--json" || std::string(argv[1]) == "--program-json"))
    {
        std::vector<unsigned char> data;
        if (read_file(argv[2], data) != 0)
        {
            fprintf(stderr, "read failed\n");
            return 1;
        }

        if (std::string(argv[1]) == "--json")
        {
            pnnx::Pt2JsonValue value;
            std::string error;
            const int result = pnnx::parse_pt2_json(data.empty() ? 0 : &data[0], data.size(), value, error);
            if (result != 0)
                fprintf(stderr, "%s\n", error.c_str());
            return result != 0;
        }

        pnnx::Pt2Program program;
        const int result = pnnx::parse_pt2_program(data.empty() ? 0 : &data[0], data.size(), program);
        if (result != 0)
            fprintf(stderr, "%s\n", program.error.c_str());
        return result != 0;
    }

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
