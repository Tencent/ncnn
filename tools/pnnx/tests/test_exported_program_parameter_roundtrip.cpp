// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ir.h"
#include "model_stat.h"
#include "storezip.h"
#include "utils.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <string>
#include <vector>

static int test_parameter_roundtrip()
{
    pnnx::Parameter original;
    original.type = 6;

    const std::string encoded = pnnx::Parameter::encode_to_string(original);
    if (encoded != "[]f")
    {
        fprintf(stderr, "empty float list encoded as %s\n", encoded.c_str());
        return 1;
    }

    const pnnx::Parameter parsed = pnnx::Parameter::parse_from_string(encoded);
    if (parsed.type != 6 || !parsed.af.empty())
    {
        fprintf(stderr, "empty float list decoded with type %d and size %lu\n", parsed.type, (unsigned long)parsed.af.size());
        return 1;
    }

    const std::string roundtrip = pnnx::Parameter::encode_to_string(parsed);
    if (roundtrip != encoded)
    {
        fprintf(stderr, "empty float list roundtrip changed %s to %s\n", encoded.c_str(), roundtrip.c_str());
        return 1;
    }

    int failures = 0;
    const pnnx::Parameter zero = pnnx::Parameter::parse_from_string(pnnx::Parameter::encode_to_string(pnnx::Parameter(-0.f)));
    if (zero.type != 3 || zero.f != 0.f || !std::signbit(zero.f))
    {
        fprintf(stderr, "negative zero scalar lost its sign\n");
        failures++;
    }

    const pnnx::Parameter zeros = pnnx::Parameter::parse_from_string(pnnx::Parameter::encode_to_string(pnnx::Parameter{-0.f, 0.f}));
    if (zeros.type != 6 || zeros.af.size() != 2 || zeros.af[0] != 0.f || !std::signbit(zeros.af[0]) || zeros.af[1] != 0.f || std::signbit(zeros.af[1]))
    {
        fprintf(stderr, "float list zeros lost their signs\n");
        failures++;
    }

    const double zero64 = strtod(pnnx::double_to_string(-0.0).c_str(), 0);
    if (zero64 != 0.0 || !std::signbit(zero64))
    {
        fprintf(stderr, "negative zero double lost its sign\n");
        failures++;
    }

    return failures;
}

static int test_attribute_contract()
{
    int failures = 0;

    const pnnx::Attribute scalar({}, {3.25f});
    const std::vector<float> scalar_data = scalar.get_float32_data();
    if (scalar.type != 1 || !scalar.shape.empty() || scalar.elemcount() != 1 || scalar.data.size() != sizeof(float) || scalar_data.size() != 1 || scalar_data[0] != 3.25f)
    {
        fprintf(stderr, "scalar attribute did not contain one float32 element\n");
        failures++;
    }

    const pnnx::Attribute vector({1}, {-2.5f});
    if (vector.shape != std::vector<int>(1, 1) || vector.elemcount() != 1 || vector.data.size() != sizeof(float))
    {
        fprintf(stderr, "rank-one attribute changed shape or element count\n");
        failures++;
    }

    const pnnx::Attribute empty({0, 2}, {});
    if (empty.shape != std::vector<int>({0, 2}) || empty.elemcount() != 0 || !empty.data.empty())
    {
        fprintf(stderr, "zero-element attribute changed shape or element count\n");
        failures++;
    }

    const pnnx::Attribute null_attribute;
    if (null_attribute.type != 0 || null_attribute.elemcount() != 0 || !null_attribute.data.empty())
    {
        fprintf(stderr, "null attribute acquired elements\n");
        failures++;
    }

    return failures;
}

static int test_attribute_payload_size_mismatch()
{
    pnnx::Attribute oversized({}, {3.25f});
    oversized.data.push_back(0);
    if (!oversized.get_float32_data().empty())
    {
        fprintf(stderr, "oversized attribute payload was presented as valid data\n");
        return 1;
    }

    const pnnx::Attribute oversized_initializer({}, {3.25f, 4.5f});
    if (!oversized_initializer.data.empty() || !oversized_initializer.get_float32_data().empty())
    {
        fprintf(stderr, "oversized attribute initializer was partially accepted\n");
        return 1;
    }

    pnnx::Attribute truncated({}, {3.25f});
    truncated.data.resize(2);
    if (!truncated.get_float32_data().empty())
    {
        fprintf(stderr, "truncated attribute payload was presented as valid data\n");
        return 1;
    }

    const int metadata_only_types[] = {1, 2, 3};
    for (int type : metadata_only_types)
    {
        pnnx::Attribute metadata_only;
        metadata_only.type = type;
        if (!metadata_only.shape.empty() || !metadata_only.data.empty() || !metadata_only.get_float32_data().empty())
        {
            fprintf(stderr, "metadata-only type %d scalar attribute was presented as valid data\n", type);
            return 1;
        }
    }

    const pnnx::Attribute empty_initializer({}, {});
    if (!empty_initializer.data.empty() || !empty_initializer.get_float32_data().empty())
    {
        fprintf(stderr, "empty scalar initializer was presented as valid data\n");
        return 1;
    }

    return 0;
}

static int test_attribute_file_roundtrip()
{
    const char* param_path = "attribute_roundtrip.pnnx.param";
    const char* bin_path = "attribute_roundtrip.pnnx.bin";
    remove(param_path);
    remove(bin_path);

    int failures = 0;
    {
        pnnx::Graph graph;
        pnnx::Operator* op = graph.new_operator("pnnx.Attribute", "attribute");
        pnnx::Operand* output = graph.new_operand("out");
        output->producer = op;
        op->outputs.push_back(output);
        op->attrs["scalar"] = pnnx::Attribute({}, {3.25f});
        op->attrs["vector"] = pnnx::Attribute({1}, {-2.5f});
        op->attrs["empty"] = pnnx::Attribute({0, 2}, {});
        op->attrs["null"] = pnnx::Attribute();

        if (graph.save(param_path, bin_path) != 0)
        {
            fprintf(stderr, "failed to save synthetic attribute graph\n");
            failures++;
        }
    }

    if (!failures)
    {
        pnnx::Graph loaded;
        if (loaded.load(param_path, bin_path) != 0 || loaded.ops.size() != 1)
        {
            fprintf(stderr, "failed to load synthetic attribute graph\n");
            failures++;
        }
        else
        {
            const pnnx::Operator* op = loaded.ops[0];
            const pnnx::Attribute& scalar = op->attrs.at("scalar");
            const pnnx::Attribute& vector = op->attrs.at("vector");
            const pnnx::Attribute& empty = op->attrs.at("empty");
            const pnnx::Attribute& null_attribute = op->attrs.at("null");
            const std::vector<float> scalar_data = scalar.get_float32_data();
            const std::vector<float> vector_data = vector.get_float32_data();
            if (scalar.type != 1 || !scalar.shape.empty() || scalar.data.size() != sizeof(float) || scalar_data.size() != 1 || scalar_data[0] != 3.25f)
            {
                fprintf(stderr, "scalar attribute file roundtrip lost shape, type, or data\n");
                failures++;
            }
            if (vector.type != 1 || vector.shape != std::vector<int>(1, 1) || vector.data.size() != sizeof(float) || vector_data.size() != 1 || vector_data[0] != -2.5f)
            {
                fprintf(stderr, "rank-one attribute file roundtrip lost shape, type, or data\n");
                failures++;
            }
            if (empty.type != 1 || empty.shape != std::vector<int>({0, 2}) || empty.elemcount() != 0 || !empty.data.empty())
            {
                fprintf(stderr, "zero-element attribute file roundtrip changed\n");
                failures++;
            }
            if (null_attribute.type != 0 || !null_attribute.shape.empty() || null_attribute.elemcount() != 0 || !null_attribute.data.empty())
            {
                fprintf(stderr, "null attribute file roundtrip changed\n");
                failures++;
            }
        }
    }

    remove(param_path);
    remove(bin_path);
    return failures;
}

static int test_explicit_scalar_parse()
{
    const std::string param = "7767517\n"
                              "1 1\n"
                              "pnnx.Attribute attribute 0 1 x @data=()f32 #x=()f32\n";

    pnnx::Graph graph;
    if (graph.parse(param) != 0 || graph.ops.size() != 1 || graph.operands.size() != 1)
    {
        fprintf(stderr, "failed to parse explicit scalar attribute and operand shape\n");
        return 1;
    }

    const pnnx::Attribute& attr = graph.ops[0]->attrs.at("data");
    const pnnx::Operand* output = graph.get_operand("x");
    if (attr.type != 1 || !attr.shape.empty() || !attr.data.empty() || !output || output->type != 1 || !output->shape.empty())
    {
        fprintf(stderr, "explicit scalar syntax did not preserve empty shape\n");
        return 1;
    }

    return 0;
}

static int check_zip(const char* path, const char* expected)
{
    pnnx::StoreZipReader reader;
    if (reader.open(path) != 0)
    {
        fprintf(stderr, "failed to open zip archive %s\n", path);
        return 1;
    }

    std::vector<std::string> names;
    if (reader.get_names(names) != 0 || names.size() != 1 || names[0] != "entry")
    {
        fprintf(stderr, "zip archive %s did not contain exactly one entry\n", path);
        return 1;
    }

    char actual[4];
    if (reader.get_file_size("entry") != sizeof(actual) || reader.read_file("entry", actual) != 0 || memcmp(actual, expected, sizeof(actual)) != 0)
    {
        fprintf(stderr, "zip archive %s contained the wrong payload\n", path);
        return 1;
    }

    return 0;
}

static int test_storezip_writer_lifecycle()
{
    const char* first_path = "storezip_first.zip";
    const char* second_path = "storezip_second.zip";
    const char* single_path = "storezip_single.zip";
    remove(first_path);
    remove(second_path);
    remove(single_path);

    int failures = 0;
    {
        pnnx::StoreZipWriter writer;
        if (writer.open(first_path) != 0 || writer.write_file("entry", "abcd", 4) != 0 || writer.close() != 0)
        {
            fprintf(stderr, "failed first writer lifecycle\n");
            failures++;
        }
        if (writer.open(second_path) != 0 || writer.write_file("entry", "efgh", 4) != 0 || writer.close() != 0)
        {
            fprintf(stderr, "failed second writer lifecycle\n");
            failures++;
        }
        if (writer.close() != 0)
        {
            fprintf(stderr, "closing an inactive writer failed\n");
            failures++;
        }
    }

    {
        pnnx::StoreZipWriter writer;
        if (writer.open(single_path) != 0 || writer.write_file("entry", "ijkl", 4) != 0 || writer.close() != 0)
        {
            fprintf(stderr, "failed single-use writer lifecycle\n");
            failures++;
        }
    }

    failures += check_zip(first_path, "abcd");
    failures += check_zip(second_path, "efgh");
    failures += check_zip(single_path, "ijkl");

    remove(first_path);
    remove(second_path);
    remove(single_path);
    return failures;
}

template<typename T>
static bool scalar_data_equals(const pnnx::Attribute& attr, int type, const T& expected)
{
    if (attr.type != type || !attr.shape.empty() || attr.elemcount() != 1 || attr.data.size() != sizeof(T))
        return false;

    T actual;
    memcpy(&actual, attr.data.data(), sizeof(actual));
    return actual == expected;
}

static int test_real_scalar_state_roundtrip(const char* param_path, const char* bin_path)
{
    pnnx::Graph graph;
    if (graph.load(param_path, bin_path) != 0)
    {
        fprintf(stderr, "failed to load real scalar-state graph\n");
        return 1;
    }

    int f32_count = 0;
    int f64_count = 0;
    int i64_count = 0;
    int bool_count = 0;
    for (const pnnx::Operator* op : graph.ops)
    {
        for (const auto& item : op->attrs)
        {
            const pnnx::Attribute& attr = item.second;
            if (scalar_data_equals(attr, 1, 3.25f))
                f32_count++;
            else if (scalar_data_equals(attr, 2, -6.5))
                f64_count++;
            else if (scalar_data_equals(attr, 5, (int64_t)1234567890123456789LL))
                i64_count++;
            else if (scalar_data_equals(attr, 9, (uint8_t)1))
                bool_count++;
        }
    }

    if (f32_count != 1 || f64_count != 1 || i64_count != 1 || bool_count != 1)
    {
        fprintf(stderr, "real scalar-state attributes did not preserve exact dtype/shape/bytes: f32=%d f64=%d i64=%d bool=%d\n", f32_count, f64_count, i64_count, bool_count);
        return 1;
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 4)
    {
        pnnx::Graph graph;
        if (graph.load(argv[1], argv[2]) != 0)
            return 1;
        return graph.python(argv[3], argv[2], {}, pnnx::get_model_stat(graph), true);
    }
    if (argc == 3)
        return test_real_scalar_state_roundtrip(argv[1], argv[2]);

    if (argc != 1)
    {
        fprintf(stderr, "usage: %s [model.pnnx.param model.pnnx.bin [model_pnnx.py]]\n", argv[0]);
        return 1;
    }

    int failures = 0;
    failures += test_parameter_roundtrip();
    failures += test_attribute_payload_size_mismatch();
    failures += test_attribute_contract();
    failures += test_storezip_writer_lifecycle();
    failures += test_attribute_file_roundtrip();
    failures += test_explicit_scalar_parse();
    return failures ? 1 : 0;
}
