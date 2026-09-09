// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <iterator>

#include "ir.h"
#include "model_stat.h"
#include "pass_ncnn/convert_half_to_float.h"

static int failures = 0;

static void expect(bool condition, const char* message)
{
    if (!condition)
    {
        fprintf(stderr, "FAILED: %s\n", message);
        failures++;
    }
}

template<typename T>
static pnnx::Attribute raw_attribute(int type, const std::vector<int>& shape, const std::vector<T>& values)
{
    pnnx::Attribute attr;
    attr.type = type;
    attr.shape = shape;
    attr.data.resize(values.size() * sizeof(T));
    if (!values.empty())
        memcpy(attr.data.data(), values.data(), attr.data.size());
    return attr;
}

static pnnx::Operand* add_attribute(pnnx::Graph& graph, const std::string& name, const pnnx::Attribute& attr)
{
    pnnx::Operator* op = graph.new_operator("pnnx.Attribute", name);
    op->attrs["data"] = attr;
    pnnx::Operand* value = graph.new_operand(name + "_value");
    value->producer = op;
    value->type = attr.type;
    value->shape = attr.shape;
    op->outputs.push_back(value);
    pnnx::Operator* output = graph.new_operator("pnnx.Output", name + "_output");
    output->inputs.push_back(value);
    value->consumers.push_back(output);
    return value;
}

static void make_fixture(pnnx::Graph& graph)
{
    pnnx::Operand* first = add_attribute(graph, "scalar_f32", raw_attribute<uint32_t>(1, {}, {0x3fa00000}));
    add_attribute(graph, "scalar_f16", raw_attribute<uint16_t>(3, {}, {0xbd00}));
    add_attribute(graph, "scalar_bf16", raw_attribute<uint16_t>(13, {}, {0x3fc0}));
    add_attribute(graph, "scalar_bool", raw_attribute<uint8_t>(9, {}, {1}));
    add_attribute(graph, "scalar_i32", raw_attribute<int32_t>(4, {}, {-7}));
    add_attribute(graph, "scalar_i64", raw_attribute<int64_t>(5, {}, {1099511627776LL}));
    add_attribute(graph, "scalar_c32", raw_attribute<uint16_t>(12, {}, {0x3e00, 0xc000}));
    add_attribute(graph, "empty_bf16", raw_attribute<uint16_t>(13, {2, 0, 3}, {}));
    add_attribute(graph, "empty_c32", raw_attribute<uint16_t>(12, {0}, {}));
    add_attribute(graph, "empty_bool", raw_attribute<uint8_t>(9, {0, 2}, {}));
    add_attribute(graph, "weight_bf16", raw_attribute<uint16_t>(13, {2, 3}, {0x3f80, 0xc000, 0x3f00, 0x4080, 0x8000, 0x40c0}));
    add_attribute(graph, "vector_c32", raw_attribute<uint16_t>(12, {2}, {0x3c00, 0x4000, 0xc200, 0x3800}));

    // Exercise integer/bool buffers on an nn.* module as well as attributes.
    pnnx::Operator* identity = graph.new_operator("nn.Identity", "identity");
    identity->inputs.push_back(first);
    first->consumers.push_back(identity);
    identity->attrs["counter"] = raw_attribute<int64_t>(5, {}, {7});
    identity->attrs["flag"] = raw_attribute<uint8_t>(9, {}, {0});
    identity->attrs["running_mean"] = raw_attribute<uint16_t>(13, {1}, {0x3f80});
    pnnx::Operand* last = graph.new_operand("identity_value");
    last->producer = identity;
    last->type = first->type;
    last->shape = first->shape;
    identity->outputs.push_back(last);
    pnnx::Operator* output = graph.new_operator("pnnx.Output", "identity_output");
    output->inputs.push_back(last);
    last->consumers.push_back(output);
}

static std::string read_text(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

static void test_counts_and_payloads()
{
    pnnx::Attribute null;
    expect(null.elemcount() == 0, "default attribute is null, not scalar");
    null.set_float32_data({1.f});
    expect(null.type == 0 && null.data.empty(), "null setter is a no-op");
    pnnx::Attribute scalar({}, {1.25f});
    expect(scalar.elemcount() == 1 && scalar.data.size() == 4, "scalar float constructor");
    expect(scalar.get_float32_data() == std::vector<float>({1.25f}), "scalar float getter");
    expect((scalar + scalar).type == 0, "scalars have no concatenation axis");
    pnnx::Attribute empty({2, 0, 3}, {});
    expect(empty.elemcount() == 0 && empty.data.empty(), "zero-sized constructor");
    pnnx::Attribute bad_constructor({2}, {1.f});
    expect(bad_constructor.data.empty(), "constructor rejects short source");

    for (int type : {
                1, 2, 3, 13
            })
    {
        pnnx::Attribute attr;
        attr.type = type;
        expect(attr.elemcount() == 1, "typed scalar count does not depend on payload");
        expect(attr.get_float32_data().empty(), "missing scalar payload is rejected");
        attr.set_float32_data({-1.25f});
        const std::vector<char> original = attr.data;
        expect(original.size() == attr.elemsize(), "scalar setter writes one item");
        expect(attr.get_float32_data() == std::vector<float>({-1.25f}), "exact scalar conversion");
        attr.set_float32_data(attr.get_float32_data());
        expect(attr.data == original, "exact scalar get/set roundtrip");
        attr.set_float32_data({});
        attr.set_float32_data({1.f, 2.f});
        expect(attr.data == original, "mismatched setter preserves payload");
        for (size_t bytes : {
                    attr.elemsize() - 1, attr.elemsize() + 1, attr.elemsize() * 2
                })
        {
            attr.data.assign(bytes, 0);
            expect(attr.get_float32_data().empty(), "getter rejects short/long/unaligned payload");
        }
        attr.set_float32_data({-1.25f});
        expect(attr.data == original, "valid setter repairs an inconsistent old payload");
        attr.shape = {0, 3};
        expect(attr.get_float32_data().empty(), "zero shape must not read stale bytes");
        attr.set_float32_data({});
        expect(attr.data.empty() && attr.elemcount() == 0, "empty setter clears payload");
        attr.set_float32_data({1.f});
        expect(attr.data.empty(), "empty shape rejects nonempty source");
    }

    for (const std::vector<int>& shape : std::vector<std::vector<int> > {{-1}, {-233, 0}, {INT_MAX, 2}, {65536, 65536}, {INT_MAX, INT_MAX, INT_MAX}})
    {
        pnnx::Attribute attr = scalar;
        attr.shape = shape;
        expect(attr.elemcount() == 0, "negative/overflowing count returns zero, never wraps");
        expect(attr.get_float32_data().empty(), "invalid shape cannot allocate/read");
        attr.set_float32_data({});
        expect(attr.data == scalar.data, "invalid shape cannot overwrite");
    }
    empty.shape = {INT_MAX, INT_MAX, 0};
    expect(empty.elemcount() == 0 && empty.get_float32_data().empty(), "zero dimension takes precedence over product overflow");
    pnnx::Attribute boolean = raw_attribute<uint8_t>(9, {}, {1});
    boolean.set_float32_data({0.f});
    expect(boolean.data == std::vector<char>({1}), "unsupported float setter preserves bool storage");
}

static void test_bf16_bits()
{
    const uint32_t input[] = {0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f808000, 0x3f818000, 0x00010000, 0x7f800000, 0xff800000, 0x7f800001};
    const uint16_t expected[] = {0x0000, 0x8000, 0x3f80, 0xbf80, 0x3f80, 0x3f82, 0x0001, 0x7f80, 0xff80, 0x7fc0};
    for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); i++)
    {
        float value;
        memcpy(&value, &input[i], sizeof(value));
        pnnx::Attribute attr;
        attr.type = 13;
        attr.set_float32_data({value});
        uint16_t actual = 0;
        if (attr.data.size() == sizeof(actual))
            memcpy(&actual, attr.data.data(), sizeof(actual));
        expect(actual == expected[i], "bf16 RNE, signed zero, subnormal, Inf and NaN bits");
        const std::vector<float> decoded = attr.get_float32_data();
        uint32_t decoded_bits = 0;
        if (decoded.size() == 1)
            memcpy(&decoded_bits, decoded.data(), sizeof(decoded_bits));
        expect(decoded.size() == 1 && decoded_bits == (uint32_t)expected[i] << 16, "bf16 exact widening bits");
    }
}

static void test_roundtrip_and_lowering()
{
    const std::string prefix = "test_ir_tensor_contract";
    pnnx::Graph graph;
    make_fixture(graph);
    // Raw serialization must also preserve subnormals and NaN payloads.
    add_attribute(graph, "f16_subnormal", raw_attribute<uint16_t>(3, {}, {0x0001}));
    add_attribute(graph, "f32_nan", raw_attribute<uint32_t>(1, {}, {0x7fc01234}));
    add_attribute(graph, "bf16_nan", raw_attribute<uint16_t>(13, {}, {0x7fc1}));
    add_attribute(graph, "bool_false", raw_attribute<uint8_t>(9, {}, {0}));
    expect(graph.save(prefix + ".param", prefix + ".bin") == 0, "save tensor IR");
    pnnx::Graph restored;
    expect(restored.load(prefix + ".param", prefix + ".bin") == 0, "load scalar and empty tensor IR");
    expect(restored.ops.size() == graph.ops.size(), "roundtrip operator count");
    if (restored.ops.size() == graph.ops.size())
    {
        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            expect(restored.ops[i]->attrs == graph.ops[i]->attrs, "exact dtype/shape/bytes roundtrip");
            for (size_t j = 0; j < graph.ops[i]->outputs.size(); j++)
            {
                const pnnx::Operand* a = graph.ops[i]->outputs[j];
                const pnnx::Operand* b = restored.ops[i]->outputs[j];
                expect(a->shape == b->shape && a->type == b->type, "scalar operand metadata survives save/load");
            }
        }
    }
    pnnx::Graph pattern;
    expect(pattern.parse(read_text(prefix + ".param")) == 0, "scalar/empty IR pattern parsing");
    expect(pattern.ops[0]->attrs.at("data").elemcount() == 1, "scalar pattern count");

    expect(graph.python(prefix + ".py", prefix + ".bin", {}, pnnx::ModelStat()) == 0, "generate dtype-aware Python");
    const std::string python = read_text(prefix + ".py");
    expect(python.find("tensor.view(torch.bfloat16)") != std::string::npos, "Python keeps bf16 via view");
    expect(python.find("tensor.view(torch.complex32)") != std::string::npos, "Python loads complex half via view");
    expect(python.find("register_buffer('scalar_bool_data'") != std::string::npos, "bool attribute is a buffer");
    expect(python.find("self.identity.register_buffer('counter'") != std::string::npos, "integer module attribute is a buffer");
    expect(python.find("np.memmap") == std::string::npos, "empty-safe loader does not mmap");

    pnnx::Graph string_graph;
    pnnx::Operator* constant = string_graph.new_operator("prim::Constant", "string_constant");
    constant->params["value"] = pnnx::Parameter("don't\\stop\n");
    pnnx::Operand* constant_value = string_graph.new_operand("string_value");
    constant_value->producer = constant;
    constant->outputs.push_back(constant_value);
    pnnx::Operator* string_output = string_graph.new_operator("pnnx.Output", "string_output");
    string_output->inputs.push_back(constant_value);
    constant_value->consumers.push_back(string_output);
    expect(string_graph.python("test_ir_string_constant.py", "test_ir_string_constant.bin", {}, pnnx::ModelStat()) == 0, "generate escaped Python string constant");
    expect(read_text("test_ir_string_constant.py").find("v_string_value = 'don\\'t\\\\stop\\n'") != std::string::npos, "Python string constants escape quotes, backslashes and newlines");
    remove("test_ir_string_constant.py");
    remove("test_ir_string_constant.bin");

    graph.ops[4]->attrs["data"].params["test_metadata"] = 42;
    pnnx::ncnn::convert_half_to_float(graph);
    for (size_t i = 0; i < graph.ops.size() && i < restored.ops.size(); i++)
    {
        for (const auto& it : restored.ops[i]->attrs)
        {
            const pnnx::Attribute& before = it.second;
            const pnnx::Attribute& after = graph.ops[i]->attrs.at(it.first);
            if (before.type == 3 || before.type == 13)
            {
                const std::vector<float> expected = before.get_float32_data();
                expect(after.type == 1 && after.shape == before.shape, "only lowering promotes half weights");
                expect(after.data.size() == expected.size() * sizeof(float), "lowered scalar/empty payload size");
                if (!expected.empty() && after.data.size() == expected.size() * sizeof(float))
                    expect(memcmp(after.data.data(), expected.data(), after.data.size()) == 0, "exact widened payload including NaN bits");
            }
            else
            {
                expect(after == before, "lowering leaves other dtypes alone");
            }
        }
    }
    expect(graph.ops[4]->attrs.at("data").params.at("test_metadata").i == 42, "lowering retains attribute metadata");
    // No graph input/output dtype promise is made by this weight-only pass.
    pnnx::Graph invalid;
    pnnx::Attribute bad = raw_attribute<uint16_t>(13, {}, {0x3f80, 0x4000});
    add_attribute(invalid, "bad", bad);
    pnnx::ncnn::convert_half_to_float(invalid);
    expect(invalid.ops[0]->attrs.at("data") == bad, "lowering refuses malformed payload without looping");

    remove((prefix + ".param").c_str());
    remove((prefix + ".bin").c_str());
    remove((prefix + ".py").c_str());
}

static void test_bad_archive_lengths()
{
    const std::string prefix = "test_ir_tensor_contract_bad";
    for (size_t size : {
                size_t(0), size_t(3), size_t(5), size_t(8)
            })
    {
        pnnx::Graph graph;
        pnnx::Attribute bad;
        bad.type = 1;
        bad.data.resize(size);
        add_attribute(graph, "bad", bad);
        expect(graph.save(prefix + ".param", prefix + ".bin") == 0, "write intentionally malformed fixture");
        pnnx::Graph restored;
        expect(restored.load(prefix + ".param", prefix + ".bin") != 0, "reject mismatched archive size before reading");
    }
    remove((prefix + ".param").c_str());
    remove((prefix + ".bin").c_str());
}

int main(int argc, char** argv)
{
    if (argc == 3 && std::string(argv[1]) == "--python-fixture")
    {
        const std::string prefix = argv[2];
        pnnx::Graph graph;
        make_fixture(graph);
        if (graph.save(prefix + ".param", prefix + ".bin") != 0)
            return 1;
        return graph.python(prefix + ".py", prefix + ".bin", {}, pnnx::ModelStat()) == 0 ? 0 : 1;
    }
    if (argc != 1)
        return 1;
    test_counts_and_payloads();
    test_bf16_bits();
    test_roundtrip_and_lowering();
    test_bad_archive_lengths();
    return failures == 0 ? 0 : 1;
}