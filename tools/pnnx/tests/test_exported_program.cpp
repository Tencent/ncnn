// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <string.h>

#include <limits>
#include <string>

#include <torch/csrc/jit/operator_upgraders/utils.h>

#include "exported_program.h"
#include "storezip.h"

static int test_failures = 0;

static void expect_true(bool value, const char* message)
{
    if (value)
        return;

    fprintf(stderr, "FAILED: %s\n", message);
    test_failures++;
}

static void expect_success(bool value, const std::string& error)
{
    expect_true(value, error.c_str());
}

static void test_defaults()
{
    pnnx::pt2::ExportedProgramArchive archive;
    expect_true(archive.archive_version == 0, "default archive version");
    expect_true(archive.program.graph.inputs.empty(), "default graph inputs");
    expect_true(!archive.program.graph.is_single_tensor_return, "default multiple return flag");

    pnnx::pt2::TensorMeta tensor_meta;
    expect_true(tensor_meta.scalar_type == 0, "default scalar type");
    expect_true(!tensor_meta.requires_grad, "default requires grad");
    expect_true(!tensor_meta.device.has_index, "default device index");
}

static void test_minimal_archive()
{
    pnnx::pt2::ExportedProgramArchive archive;
    archive.model_name = "model";
    archive.program.schema_version.major = 8;
    archive.program.schema_version.minor = 14;
    archive.program.opset_version["aten"] = 10;

    pnnx::pt2::SymInt batch;
    batch.type = pnnx::pt2::SymInt::Expression;
    batch.expression = "s0";
    batch.has_hint = true;
    batch.hint = 2;

    pnnx::pt2::SymInt features;
    features.integer = 16;

    pnnx::pt2::TensorMeta input_meta;
    input_meta.scalar_type = 7;
    input_meta.sizes.push_back(batch);
    input_meta.sizes.push_back(features);
    archive.program.graph.tensor_values["input"] = input_meta;

    pnnx::pt2::InputSpec input;
    input.type = pnnx::pt2::InputSpec::UserInput;
    input.argument.type = pnnx::pt2::Argument::Tensor;
    input.argument.name = "input";
    archive.program.signature.inputs.push_back(input);

    pnnx::pt2::InputSpec weight;
    weight.type = pnnx::pt2::InputSpec::Parameter;
    weight.argument.type = pnnx::pt2::Argument::Tensor;
    weight.argument.name = "weight";
    weight.target = "linear.weight";
    archive.program.signature.inputs.push_back(weight);

    pnnx::pt2::RangeConstraint constraint;
    constraint.has_min = true;
    constraint.min = 1;
    constraint.has_max = true;
    constraint.max = 8;
    archive.program.range_constraints["s0"] = constraint;

    pnnx::pt2::PayloadMeta payload;
    payload.path = "weight_0";
    payload.is_parameter = true;
    payload.has_tensor_meta = true;
    archive.state_dict["linear.weight"] = payload;

    expect_true(archive.program.graph.tensor_values["input"].sizes[0].hint == 2, "symbolic dimension hint");
    expect_true(archive.program.signature.inputs[1].target == "linear.weight", "parameter target");
    expect_true(archive.program.range_constraints["s0"].max == 8, "range constraint");
    expect_true(archive.state_dict["linear.weight"].is_parameter, "parameter payload");
}

// clang-format off
static const char* exported_program_json = R"json({
    "graph_module": {
        "graph": {
            "inputs": [{"as_tensor":{"name":"p_linear_weight"}},{"as_tensor":{"name":"x"}}],
            "outputs": [{"as_tensor":{"name":"linear"}}],
            "nodes": [{
                "name": "linear",
                "target": "torch.ops.aten.linear.default",
                "inputs": [
                    {"name":"input","arg":{"as_tensor":{"name":"x"}},"kind":1},
                    {"name":"weight","arg":{"as_tensor":{"name":"p_linear_weight"}},"kind":1}
                ],
                "outputs": [{"as_tensor":{"name":"linear"}}],
                "metadata": {"torch_fn":"linear"},
                "future_node_field": true
            }],
            "tensor_values": {
                "p_linear_weight":{"dtype":7,"sizes":[{"as_int":4},{"as_int":3}],"requires_grad":true,"device":{"type":"cpu","index":null},"strides":[{"as_int":3},{"as_int":1}],"storage_offset":{"as_int":0},"layout":7},
                "x":{"dtype":7,"sizes":[{"as_expr":{"expr_str":"s0","hint":{"as_int":2}}},{"as_int":3}],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":3},{"as_int":1}],"storage_offset":{"as_int":0},"layout":7},
                "linear":{"dtype":7,"sizes":[{"as_expr":{"expr_str":"s0","hint":{"as_int":2}}},{"as_int":4}],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":4},{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}
            },
            "sym_int_values": {},
            "sym_bool_values": {},
            "sym_float_values": {},
            "is_single_tensor_return": true
        },
        "signature": {
            "input_specs": [
                {"parameter":{"arg":{"name":"p_linear_weight"},"parameter_name":"linear.weight"}},
                {"user_input":{"arg":{"as_tensor":{"name":"x"}}}}
            ],
            "output_specs": [{"user_output":{"arg":{"as_tensor":{"name":"linear"}}}}]
        },
        "module_call_graph": [],
        "future_graph_module_field": "ignored"
    },
    "opset_version": {"aten": 10},
    "range_constraints": {"s0":{"min_val":1,"max_val":null}},
    "schema_version": {"major":8,"minor":20},
    "torch_version": "2.12.0",
    "future_exported_program_field": {"ignored":true}
})json";
// clang-format on

static void test_parse_exported_program()
{
    pnnx::pt2::ExportedProgram program;
    std::string error;
    expect_success(pnnx::pt2::parse_exported_program(exported_program_json, program, error), error);
    expect_true(program.schema_version.major == 8 && program.schema_version.minor == 20, "schema version");
    expect_true(program.graph.nodes.size() == 1 && program.graph.nodes[0].name == "linear", "graph node");
    expect_true(program.graph.nodes[0].inputs.size() == 2 && program.graph.nodes[0].inputs[0].kind == pnnx::pt2::NamedArgument::Positional, "named positional arguments");
    expect_true(program.signature.inputs.size() == 2 && program.signature.inputs[0].target == "linear.weight", "graph signature");
    expect_true(program.graph.tensor_values["x"].sizes[0].has_hint && program.graph.tensor_values["x"].sizes[0].hint == 2, "symbolic shape hint");
    expect_true(program.range_constraints["s0"].has_min && !program.range_constraints["s0"].has_max, "unbounded range constraint");
}

static void test_concrete_symbolic_arguments()
{
    // clang-format off
    const char* document = R"json({"graph_module":{"graph":{"inputs":[],"outputs":[],"nodes":[{"name":"symbols","target":"torch.ops.aten.symbols.default","inputs":[{"name":"predicate","arg":{"as_sym_bool":{"as_bool":true}},"kind":1},{"name":"scale","arg":{"as_sym_float":{"as_float":1.5}},"kind":1},{"name":"limit","arg":{"as_sym_float":{"as_float":"-Infinity"}},"kind":1}],"outputs":[],"metadata":{}}],"tensor_values":{},"sym_int_values":{}} ,"signature":{"input_specs":[],"output_specs":[]}},"opset_version":{"aten":10},"range_constraints":{},"schema_version":{"major":8,"minor":20}})json";
    // clang-format on

    pnnx::pt2::ExportedProgram program;
    std::string error;
    expect_success(pnnx::pt2::parse_exported_program(document, program, error), error);
    expect_true(program.graph.nodes.size() == 1 && program.graph.nodes[0].inputs.size() == 3, "concrete symbolic arguments");
    if (program.graph.nodes.size() == 1 && program.graph.nodes[0].inputs.size() == 3)
    {
        const std::vector<pnnx::pt2::NamedArgument>& inputs = program.graph.nodes[0].inputs;
        expect_true(inputs[0].argument.type == pnnx::pt2::Argument::SymBoolean && inputs[0].argument.boolean, "concrete symbolic boolean");
        expect_true(inputs[1].argument.type == pnnx::pt2::Argument::SymFloat && inputs[1].argument.floating_point == 1.5, "concrete symbolic float");
        expect_true(inputs[2].argument.type == pnnx::pt2::Argument::SymFloat && inputs[2].argument.floating_point == -std::numeric_limits<double>::infinity(), "concrete non-finite symbolic float");
    }
}

static void test_load_archive_metadata()
{
    const char* path = "test_exported_program_metadata.pt2";
    pnnx::StoreZipWriter writer;
    writer.open(path);
    writer.write_file("package/archive_format", "pt2", 3);
    writer.write_file("package/archive_version", "0", 1);
    writer.write_file("package/models/model.json", exported_program_json, std::string(exported_program_json).size());
    writer.close();

    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_success(pnnx::pt2::load_exported_program_archive_metadata(path, archive, error), error);
    expect_true(archive.model_name == "model", "archive model name");
    expect_true(archive.program.graph.nodes[0].target == "torch.ops.aten.linear.default", "archive graph target");
    remove(path);
}

static void write_payload_config(pnnx::StoreZipWriter& writer, const char* name, const char* path, bool is_parameter, const char* sizes, const char* strides, int storage_offset = 0, bool use_pickle = false, int dtype = 7, const char* device = "cpu", int layout = 7)
{
    const std::string config = std::string("{\"config\":{\"tensor\":{\"path_name\":\"") + path
                               + "\",\"is_param\":" + (is_parameter ? "true" : "false")
                               + ",\"use_pickle\":" + (use_pickle ? "true" : "false")
                               + ",\"tensor_meta\":{\"dtype\":" + std::to_string(dtype) + ",\"sizes\":" + sizes
                               + ",\"requires_grad\":false,\"device\":{\"type\":\"" + device + "\"},\"strides\":" + strides
                               + ",\"storage_offset\":{\"as_int\":" + std::to_string(storage_offset) + "},\"layout\":" + std::to_string(layout) + "}}}}";
    writer.write_file(name, config.data(), config.size());
}

static void write_empty_payload_config(pnnx::StoreZipWriter& writer, const char* name)
{
    const char* config = "{\"config\":{}}";
    writer.write_file(name, config, std::string(config).size());
}

static void write_payload_archive(const char* path, const char* payload_path, const char* sizes, const char* strides, int storage_offset, const std::vector<char>& storage, bool use_pickle = false, int dtype = 7, const char* device = "cpu", int layout = 7, const char* byteorder = 0)
{
    pnnx::StoreZipWriter writer;
    writer.open(path);
    writer.write_file("package/archive_format", "pt2", 3);
    writer.write_file("package/archive_version", "0", 1);
    if (byteorder)
        writer.write_file("package/byteorder", byteorder, strlen(byteorder));
    writer.write_file("package/models/model.json", exported_program_json, std::string(exported_program_json).size());
    write_payload_config(writer, "package/data/weights/model_weights_config.json", payload_path, true, sizes, strides, storage_offset, use_pickle, dtype, device, layout);
    write_empty_payload_config(writer, "package/data/constants/model_constants_config.json");
    writer.write_file(std::string("package/data/weights/") + payload_path, storage.data(), storage.size());
    writer.close();
}

static void test_load_tensor_payloads()
{
    const char* path = "test_exported_program_payload.pt2";
    const std::vector<char> storage(32, 42);
    write_payload_archive(path, "weight_0", "[{\"as_int\":2},{\"as_int\":2}]", "[{\"as_int\":3},{\"as_int\":1}]", 1, storage);

    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_success(pnnx::pt2::load_exported_program_archive(path, archive, error), error);
    expect_true(archive.state_dict["tensor"].is_parameter, "parameter payload metadata");
    expect_true(archive.state_dict["tensor"].tensor_meta.storage_offset.integer == 1, "payload storage offset");
    expect_true(archive.state_dict_storages["data/weights/weight_0"].size() == storage.size(), "raw payload bytes");
    remove(path);
}

static void test_shared_storage_payloads()
{
    const char* path = "test_exported_program_shared_storage.pt2";
    const char* config = "{\"config\":{"
                         "\"first\":{\"path_name\":\"weight_0\",\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":{\"dtype\":7,\"sizes\":[{\"as_int\":2}],\"requires_grad\":false,\"device\":{\"type\":\"cpu\"},\"strides\":[{\"as_int\":1}],\"storage_offset\":{\"as_int\":0},\"layout\":7}},"
                         "\"second\":{\"path_name\":\"weight_0\",\"is_param\":true,\"use_pickle\":false,\"tensor_meta\":{\"dtype\":7,\"sizes\":[{\"as_int\":2}],\"requires_grad\":false,\"device\":{\"type\":\"cpu\"},\"strides\":[{\"as_int\":1}],\"storage_offset\":{\"as_int\":2},\"layout\":7}}}}";
    const std::vector<char> storage(16, 1);

    pnnx::StoreZipWriter writer;
    writer.open(path);
    writer.write_file("package/archive_format", "pt2", 3);
    writer.write_file("package/archive_version", "0", 1);
    writer.write_file("package/models/model.json", exported_program_json, std::string(exported_program_json).size());
    writer.write_file("package/data/weights/model_weights_config.json", config, std::string(config).size());
    write_empty_payload_config(writer, "package/data/constants/model_constants_config.json");
    writer.write_file("package/data/weights/weight_0", storage.data(), storage.size());
    writer.close();

    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_success(pnnx::pt2::load_exported_program_archive(path, archive, error), error);
    expect_true(archive.state_dict_storages.size() == 1, "shared storage is loaded once");
    expect_true(archive.state_dict["first"].path == archive.state_dict["second"].path, "shared storage path is preserved");
    remove(path);
}

static void test_invalid_tensor_payloads()
{
    const std::vector<char> storage(16, 0);
    const char* out_of_bounds = "test_exported_program_payload_out_of_bounds.pt2";
    write_payload_archive(out_of_bounds, "weight_0", "[{\"as_int\":2},{\"as_int\":2}]", "[{\"as_int\":2},{\"as_int\":1}]", 1, storage);

    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_true(!pnnx::pt2::load_exported_program_archive(out_of_bounds, archive, error), "out of bounds tensor is rejected");
    expect_true(error.find("exceeds storage") != std::string::npos, "out of bounds error is explicit");
    remove(out_of_bounds);

    const char* pickled = "test_exported_program_pickled_payload.pt2";
    write_payload_archive(pickled, "weight_0", "[{\"as_int\":1}]", "[{\"as_int\":1}]", 0, storage, true);
    archive = pnnx::pt2::ExportedProgramArchive();
    expect_true(!pnnx::pt2::load_exported_program_archive(pickled, archive, error), "pickled tensor is rejected");
    expect_true(error.find("pickled") != std::string::npos, "pickled tensor error is explicit");
    remove(pickled);
}

static void test_constant_and_empty_payloads()
{
    const char* path = "test_exported_program_constant_empty.pt2";
    pnnx::StoreZipWriter writer;
    writer.open(path);
    writer.write_file("package/archive_format", "pt2", 3);
    writer.write_file("package/archive_version", "0", 1);
    writer.write_file("package/models/model.json", exported_program_json, std::string(exported_program_json).size());
    write_empty_payload_config(writer, "package/data/weights/model_weights_config.json");
    write_payload_config(writer, "package/data/constants/model_constants_config.json", "tensor_0", false, "[{\"as_int\":0},{\"as_int\":4}]", "[{\"as_int\":4},{\"as_int\":1}]");
    writer.write_file("package/data/constants/tensor_0", 0, 0);
    writer.close();

    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_success(pnnx::pt2::load_exported_program_archive(path, archive, error), error);
    expect_true(archive.constants.size() == 1 && !archive.constants["tensor"].is_parameter, "constant payload metadata");
    expect_true(archive.constant_storages["data/constants/tensor_0"].empty(), "empty tensor storage");
    remove(path);
}

static void test_missing_and_invalid_payloads()
{
    const char* missing_path = "test_exported_program_missing_payload.pt2";
    pnnx::StoreZipWriter writer;
    writer.open(missing_path);
    writer.write_file("package/archive_format", "pt2", 3);
    writer.write_file("package/archive_version", "0", 1);
    writer.write_file("package/models/model.json", exported_program_json, std::string(exported_program_json).size());
    write_payload_config(writer, "package/data/weights/model_weights_config.json", "missing", true, "[{\"as_int\":1}]", "[{\"as_int\":1}]");
    write_empty_payload_config(writer, "package/data/constants/model_constants_config.json");
    writer.close();

    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_true(!pnnx::pt2::load_exported_program_archive(missing_path, archive, error), "missing payload is rejected");
    expect_true(error.find("missing tensor payload") != std::string::npos, "missing payload error is explicit");
    remove(missing_path);

    const char* rank_path = "test_exported_program_rank_mismatch.pt2";
    write_payload_archive(rank_path, "weight_0", "[{\"as_int\":2},{\"as_int\":2}]", "[{\"as_int\":1}]", 0, std::vector<char>(16));
    expect_true(!pnnx::pt2::load_exported_program_archive(rank_path, archive, error), "rank mismatch is rejected");
    expect_true(error.find("rank mismatch") != std::string::npos, "rank mismatch error is explicit");
    remove(rank_path);

    const char* path_traversal = "test_exported_program_path_traversal.pt2";
    write_payload_archive(path_traversal, "../weight_0", "[{\"as_int\":1}]", "[{\"as_int\":1}]", 0, std::vector<char>(4));
    expect_true(!pnnx::pt2::load_exported_program_archive(path_traversal, archive, error), "payload path traversal is rejected");
    expect_true(error.find("invalid payload path") != std::string::npos, "invalid payload path error is explicit");
    remove(path_traversal);
}

static void test_tensor_range_overflow()
{
    const char* path = "test_exported_program_payload_overflow.pt2";
    const char* maximum = "9223372036854775807";
    const std::string sizes = std::string("[{\"as_int\":") + maximum + "}]";
    const std::string strides = std::string("[{\"as_int\":") + maximum + "}]";
    write_payload_archive(path, "weight_0", sizes.c_str(), strides.c_str(), 0, std::vector<char>(4));

    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_true(!pnnx::pt2::load_exported_program_archive(path, archive, error), "tensor range overflow is rejected");
    expect_true(error.find("overflows uint64") != std::string::npos, "tensor overflow error is explicit");
    remove(path);
}

static void test_invalid_schema()
{
    std::string document(exported_program_json);
    const size_t position = document.find("\"major\":8");
    document.replace(position, 9, "\"major\":9");

    pnnx::pt2::ExportedProgram program;
    std::string error;
    expect_true(!pnnx::pt2::parse_exported_program(document, program, error), "unsupported schema is rejected");
    expect_true(error.find("schema_version") != std::string::npos, "schema error has field path");
}

static void test_argument_variants()
{
    // clang-format off
    const char* document = R"json({"graph_module":{"graph":{"inputs":[],"outputs":[],"nodes":[{"target":"torch.ops.aten.index.Tensor","inputs":[{"name":"indices","arg":{"as_optional_tensors":[{"as_none":true},{"as_tensor":{"name":"index"}}]},"kind":1}],"outputs":[],"metadata":{}}],"tensor_values":{},"sym_int_values":{}} ,"signature":{"input_specs":[],"output_specs":[]}},"opset_version":{"aten":10},"range_constraints":{},"schema_version":{"major":8,"minor":20}})json";
    // clang-format on
    pnnx::pt2::ExportedProgram program;
    std::string error;
    expect_success(pnnx::pt2::parse_exported_program(document, program, error), error);
    expect_true(program.graph.nodes[0].inputs[0].argument.type == pnnx::pt2::Argument::OptionalTensors, "optional tensor list variant");
    expect_true(program.graph.nodes[0].inputs[0].argument.values[0].type == pnnx::pt2::Argument::None, "optional none variant");
    expect_true(program.graph.nodes[0].inputs[0].argument.values[1].name == "index", "optional tensor reference");
}

static void test_multiple_models_are_rejected()
{
    const char* path = "test_exported_program_multiple_models.pt2";
    pnnx::StoreZipWriter writer;
    writer.open(path);
    writer.write_file("package/archive_format", "pt2", 3);
    writer.write_file("package/archive_version", "0", 1);
    writer.write_file("package/models/model.json", exported_program_json, std::string(exported_program_json).size());
    writer.write_file("package/models/another.json", exported_program_json, std::string(exported_program_json).size());
    writer.close();

    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_true(!pnnx::pt2::load_exported_program_archive_metadata(path, archive, error), "multiple models are rejected");
    expect_true(error.find("exactly one") != std::string::npos, "multiple model error is explicit");
    remove(path);
}

static void test_raw_metadata_contract()
{
    const char* path = "test_exported_program_raw_contract.pt2";
    struct Case
    {
        int dtype;
        const char* device;
        int layout;
        const char* diagnostic;
    };
    const Case cases[] = {
        {0, "cpu", 7, "scalar type 0"},
        {14, "cpu", 7, "scalar type 14"},
        {28, "cpu", 7, "scalar type 28"},
        {29, "cpu", 7, "scalar type 29"},
        {35, "cpu", 7, "scalar type 35"},
        {999, "cpu", 7, "scalar type 999"},
        {7, "cuda", 7, "CPU device"},
        {7, "meta", 7, "CPU device"},
        {7, "cpu", 0, "Strided layout"},
        {7, "cpu", 1, "Strided layout"},
        {7, "cpu", 999, "Strided layout"}
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        write_payload_archive(path, "weight_0", "[{\"as_int\":1}]", "[{\"as_int\":1}]", 0, std::vector<char>(16), false, cases[i].dtype, cases[i].device, cases[i].layout);
        pnnx::pt2::ExportedProgramArchive archive;
        std::string error;
        expect_true(!pnnx::pt2::load_exported_program_archive(path, archive, error), "invalid raw tensor metadata rejected");
        expect_true(error.find(cases[i].diagnostic) != std::string::npos, error.c_str());
        expect_true(archive.state_dict_storages.empty(), "metadata rejected before allocating raw storage");
        remove(path);
    }
    write_payload_archive(path, "weight_0", "[{\"as_int\":2}]", "[{\"as_int\":1}]", 0, std::vector<char>(4), false, 13);
    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_success(pnnx::pt2::load_exported_program_archive(path, archive, error), error);
    expect_true(archive.state_dict["tensor"].tensor_meta.scalar_type == 13 && archive.state_dict_storages["data/weights/weight_0"].size() == 4, "reader supports raw bf16 with two-byte elements");
    remove(path);

    write_payload_archive(path, "weight_0", "[{\"as_int\":2147483647}]", "[{\"as_int\":0}]", 0, std::vector<char>(4));
    expect_true(!pnnx::pt2::load_exported_program_archive(path, archive, error), "reader rejects enormous broadcast expansion");
    expect_true(error.find("materialization budget") != std::string::npos && archive.state_dict_storages.empty(), "broadcast budget checked before storage allocation");
    remove(path);
}

static void test_version_contract()
{
    pnnx::pt2::ExportedProgram program;
    std::string error;
    const std::string fixture(exported_program_json);
    const int unsupported_minor[] = {0, 14, 19, 21, 999};
    for (size_t i = 0; i < sizeof(unsupported_minor) / sizeof(unsupported_minor[0]); i++)
    {
        std::string document = fixture;
        const std::string old = "\"minor\":20";
        document.replace(document.find(old), old.size(), "\"minor\":" + std::to_string(unsupported_minor[i]));
        expect_true(!pnnx::pt2::parse_exported_program(document, program, error), "unverified schema minor is not claimed compatible");
        expect_true(error.find("supported schema is 8.20") != std::string::npos, "precise schema minor diagnostic");
    }
    const int current = (int)torch::jit::getMaxOperatorVersion();
    const int unsupported_opset[] = {0, current - 1, current + 1};
    for (size_t i = 0; i < 3; i++)
    {
        std::string document = fixture;
        const std::string old = "\"aten\": 10";
        document.replace(document.find(old), old.size(), "\"aten\": " + std::to_string(unsupported_opset[i]));
        expect_true(!pnnx::pt2::parse_exported_program(document, program, error), "opset without exact linked operator semantics rejected");
        expect_true(error.find("opset_version.aten") != std::string::npos, "opset diagnostic identifies namespace");
    }
    std::string document = fixture;
    const std::string old = "\"aten\": 10";
    document.replace(document.find(old), old.size(), "");
    expect_true(!pnnx::pt2::parse_exported_program(document, program, error) && error.find("got missing") != std::string::npos, "missing ATen contract rejected");
    document = fixture;
    document.replace(document.find(old), old.size(), "\"aten\": " + std::to_string(current) + ", \"custom\": 1");
    expect_true(!pnnx::pt2::parse_exported_program(document, program, error) && error.find("opset_version.custom") != std::string::npos, "unknown namespace version contract rejected");
}

static std::string json_string(const std::string& value)
{
    std::string result = "\"";
    for (size_t i = 0; i < value.size(); i++)
    {
        if (value[i] == '\\' || value[i] == '"') result += '\\';
        result += value[i];
    }
    return result + "\"";
}

static std::string tree_node(const char* type, const char* context, const std::string& children)
{
    return "{\"type\":" + json_string(type) + ",\"context\":" + json_string(context) + ",\"children_spec\":[" + children + "]}";
}

static std::string with_call_graph(const std::string& calls)
{
    std::string document = exported_program_json;
    const std::string marker = "\"module_call_graph\": []";
    document.replace(document.find(marker), marker.size(), "\"module_call_graph\": " + calls);
    return document;
}

static std::string root_call(const std::string& inputs, const std::string& outputs, int protocol = 1)
{
    return "{\"fqn\":\"\",\"signature\":{\"inputs\":[],\"outputs\":[],\"in_spec\":"
           + json_string("[" + std::to_string(protocol) + "," + inputs + "]")
           + ",\"out_spec\":" + json_string("[1," + outputs + "]") + "}}";
}

static void test_call_spec_contract()
{
    const std::string leaf = "{\"type\":null,\"context\":null,\"children_spec\":[]}";
    const std::string positional = tree_node("builtins.tuple", "null", leaf);
    const std::string kwargs = tree_node("builtins.dict", "[]", "");
    const std::string inputs = tree_node("builtins.tuple", "null", positional + "," + kwargs);
    const std::string singleton = tree_node("builtins.tuple", "null", leaf);
    pnnx::pt2::ExportedProgram program;
    std::string error;
    expect_success(pnnx::pt2::parse_exported_program(with_call_graph("[" + root_call(inputs, leaf) + "]"), program, error), error);
    expect_success(pnnx::pt2::parse_exported_program(with_call_graph("[" + root_call(inputs, singleton) + "]"), program, error), error);
    // Whitespace in the JSON-encoded tuple/dict context is semantically valid.
    const std::string spaced = tree_node("builtins.tuple", " null ", positional + "," + tree_node("builtins.dict", "[ ]", ""));
    expect_success(pnnx::pt2::parse_exported_program(with_call_graph("[" + root_call(spaced, leaf) + "]"), program, error), error);

    const std::string bad_inputs[] = {
        tree_node("builtins.tuple", "null", tree_node("builtins.tuple", "null", "") + "," + tree_node("builtins.dict", "[\"x\"]", leaf)),
        tree_node("builtins.tuple", "null", tree_node("builtins.tuple", "null", singleton) + "," + kwargs),
        tree_node("builtins.tuple", "null", tree_node("builtins.tuple", "null", tree_node("builtins.dict", "[\"x\"]", leaf)) + "," + kwargs),
        tree_node("builtins.tuple", "null", positional + "," + tree_node("builtins.dict", "[\"x\"]", "")),
        tree_node("builtins.tuple", "null", tree_node("builtins.tuple", "null", leaf + "," + leaf) + "," + kwargs),
        leaf
    };
    for (size_t i = 0; i < sizeof(bad_inputs) / sizeof(bad_inputs[0]); i++)
    {
        expect_true(!pnnx::pt2::parse_exported_program(with_call_graph("[" + root_call(bad_inputs[i], leaf) + "]"), program, error), "unsupported input tree rejected");
        expect_true(error.find("in_spec") != std::string::npos && error.find("PyTree") != std::string::npos, error.c_str());
        expect_true(program.graph.inputs.empty(), "failed parse clears previous program");
    }
    const std::string bad_outputs[] = {
        tree_node("builtins.dict", "[\"x\"]", leaf),
        tree_node("builtins.list", "null", leaf),
        tree_node("collections.namedtuple", "Point", leaf),
        tree_node("builtins.tuple", "null", singleton),
        tree_node("builtins.tuple", "null", ""),
        tree_node("builtins.tuple", "null", leaf + "," + leaf),
        "{\"type\":null,\"context\":\"null\",\"children_spec\":[]}",
        "{\"type\":null,\"context\":null}",
        "{\"type\":\"builtins.tuple\",\"context\":null,\"children_spec\":[]}"
    };
    for (size_t i = 0; i < sizeof(bad_outputs) / sizeof(bad_outputs[0]); i++)
    {
        expect_true(!pnnx::pt2::parse_exported_program(with_call_graph("[" + root_call(inputs, bad_outputs[i]) + "]"), program, error), "unsupported output tree rejected");
        expect_true(error.find("out_spec") != std::string::npos && error.find("PyTree") != std::string::npos, error.c_str());
    }
    expect_true(!pnnx::pt2::parse_exported_program(with_call_graph("[" + root_call(inputs, leaf, 2) + "]"), program, error)
                && error.find("protocol") != std::string::npos, "unknown TreeSpec protocol rejected");
    expect_true(!pnnx::pt2::parse_exported_program(with_call_graph("[" + root_call("not JSON", leaf) + "]"), program, error)
                && error.find("TreeSpec JSON") != std::string::npos, "malformed inner JSON rejected");
    const std::string bad_calls[] = {"null", "[{\"fqn\":\"child\",\"signature\":null}]", "[{\"fqn\":\"\",\"signature\":null}]",
                                     "[" + root_call(inputs, leaf) + "," + root_call(inputs, leaf) + "]"};
    for (size_t i = 0; i < sizeof(bad_calls) / sizeof(bad_calls[0]); i++)
        expect_true(!pnnx::pt2::parse_exported_program(with_call_graph(bad_calls[i]), program, error)
                    && error.find("module_call_graph") != std::string::npos, "malformed root call graph rejected");

    // A flat tuple may contain tensors and numeric scalar leaves. Preserve the
    // graph/signature output order; this is not support for general PyTrees.
    std::string document = with_call_graph("[" + root_call(inputs, tree_node("builtins.tuple", "null", leaf + "," + leaf + "," + leaf + "," + leaf)) + "]");
    const std::string old_outputs = "\"outputs\": [{\"as_tensor\":{\"name\":\"linear\"}}]";
    document.replace(document.find(old_outputs), old_outputs.size(), "\"outputs\":[{\"as_tensor\":{\"name\":\"linear\"}},{\"as_int\":3},{\"as_float\":1.5},{\"as_bool\":true}]");
    const std::string old_specs = "\"output_specs\": [{\"user_output\":{\"arg\":{\"as_tensor\":{\"name\":\"linear\"}}}}]";
    const std::string new_specs = "\"output_specs\":[{\"user_output\":{\"arg\":{\"as_tensor\":{\"name\":\"linear\"}}}},{\"user_output\":{\"arg\":{\"as_int\":3}}},{\"user_output\":{\"arg\":{\"as_float\":1.5}}},{\"user_output\":{\"arg\":{\"as_bool\":true}}}]";
    document.replace(document.find(old_specs), old_specs.size(), new_specs);
    expect_success(pnnx::pt2::parse_exported_program(document, program, error), error);
}

static void test_byteorder_contract()
{
    const char* path = "test_exported_program_byteorder.pt2";
    const char* byteorders[] = {0, "little", "big", "unknown", "", "little\n"};
    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    for (size_t i = 0; i < sizeof(byteorders) / sizeof(byteorders[0]); i++)
    {
        write_payload_archive(path, "weight_0", "[{\"as_int\":1}]", "[{\"as_int\":1}]", 0, std::vector<char>(4), false, 7, "cpu", 7, byteorders[i]);
        const bool loaded = pnnx::pt2::load_exported_program_archive(path, archive, error);
        expect_true(loaded == (i < 2), "little/default accepted, big/unknown byteorder rejected");
        if (loaded)
            expect_true(archive.byteorder == "little", "resolved archive byteorder");
        else
        {
            expect_true(error.find("byteorder") != std::string::npos, error.c_str());
            expect_true(archive.state_dict_storages.empty() && archive.program.graph.nodes.empty(), "byteorder failure clears prior result");
            expect_true(!pnnx::pt2::load_exported_program_archive_metadata(path, archive, error), "metadata boundary checks byteorder too");
        }
        remove(path);
    }
}

static void test_aggregate_payload_budget()
{
    const char* path = "test_exported_program_aggregate_budget.pt2";
    pnnx::StoreZipWriter writer;
    writer.open(path);
    writer.write_file("package/archive_format", "pt2", 3);
    writer.write_file("package/archive_version", "0", 1);
    writer.write_file("package/models/model.json", exported_program_json, strlen(exported_program_json));
    // Each view is within the existing per-tensor 512 MiB bound and has a
    // four-byte source. Their combined dense size exceeds the model budget.
    write_payload_config(writer, "package/data/weights/model_weights_config.json", "weight_0", true, "[{\"as_int\":67108864}]", "[{\"as_int\":0}]");
    write_payload_config(writer, "package/data/constants/model_constants_config.json", "tensor_0", false, "[{\"as_int\":67108864}]", "[{\"as_int\":0}]");
    writer.write_file("package/data/weights/weight_0", "abcd", 4);
    writer.write_file("package/data/constants/tensor_0", "abcd", 4);
    writer.close();
    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    expect_true(!pnnx::pt2::load_exported_program_archive(path, archive, error), "cross-dictionary aggregate budget enforced");
    expect_true(error.find("aggregate 512 MiB") != std::string::npos, error.c_str());
    expect_true(archive.state_dict_storages.empty() && archive.constant_storages.empty(), "budget failure retains no partial payloads");
    remove(path);
}

// Patch only our tiny ZIP64 fixtures, matching a full record name in a fixed
// local/central header. No large/sparse archive is needed for rejection tests.
static void patch_payload_headers(const char* path, const char* name, bool compression, bool empty_crc)
{
    FILE* fp = fopen(path, "r+b");
    expect_true(fp != 0, "open fixture for header mutation");
    if (!fp) return;
    fseek(fp, 0, SEEK_END);
    const long length = ftell(fp);
    expect_true(length > 0 && length < 65536, "small ZIP mutation fixture");
    if (length <= 0 || length >= 65536) { fclose(fp); return; }
    std::vector<unsigned char> bytes((size_t)length);
    rewind(fp);
    expect_true(fread(bytes.data(), 1, bytes.size(), fp) == bytes.size(), "read ZIP mutation fixture");
    size_t patched = 0;
    for (size_t i = 0; i + 46 + strlen(name) <= bytes.size(); i++)
    {
        const bool local = memcmp(bytes.data() + i, "PK\003\004", 4) == 0;
        const bool central = memcmp(bytes.data() + i, "PK\001\002", 4) == 0;
        if (!local && !central) continue;
        const size_t header_size = local ? 30 : 46;
        if (memcmp(bytes.data() + i + header_size, name, strlen(name)) != 0) continue;
        if (compression) bytes[i + (local ? 8 : 10)] = 8;
        if (empty_crc) bytes[i + (local ? 14 : 16)] ^= 1;
        patched++;
    }
    expect_true(patched == 2, "patched matching local and central record fields");
    rewind(fp);
    expect_true(fwrite(bytes.data(), 1, bytes.size(), fp) == bytes.size(), "write ZIP mutation fixture");
    fclose(fp);
}

static void test_record_preflight_and_empty_crc()
{
    const char* path = "test_exported_program_record_features.pt2";
    pnnx::pt2::ExportedProgramArchive archive;
    std::string error;
    for (int empty = 0; empty < 2; empty++)
    {
        write_payload_archive(path, "weight_0", empty ? "[{\"as_int\":0}]" : "[{\"as_int\":1}]", "[{\"as_int\":1}]", 0, std::vector<char>(empty ? 0 : 4));
        patch_payload_headers(path, "package/data/weights/weight_0", true, false);
        expect_true(!pnnx::pt2::load_exported_program_archive(path, archive, error), "compressed payload rejected even if empty");
        expect_true(error.find("unsupported ZIP compression") != std::string::npos, error.c_str());
        expect_true(archive.state_dict_storages.empty(), "compressed payload has no retained allocation");
        remove(path);
    }
    write_payload_archive(path, "weight_0", "[{\"as_int\":0}]", "[{\"as_int\":1}]", 0, std::vector<char>());
    patch_payload_headers(path, "package/data/weights/weight_0", false, true);
    expect_true(!pnnx::pt2::load_exported_program_archive(path, archive, error), "empty record CRC is checked");
    expect_true(error.find("failed to read tensor payload") != std::string::npos, error.c_str());
    expect_true(archive.state_dict.empty() && archive.state_dict_storages.empty(), "late CRC failure clears metadata and payloads");
    remove(path);
}

int main()
{
    test_call_spec_contract();
    test_byteorder_contract();
    test_aggregate_payload_budget();
    test_record_preflight_and_empty_crc();
    test_raw_metadata_contract();
    test_version_contract();
    test_defaults();
    test_minimal_archive();
    test_parse_exported_program();
    test_concrete_symbolic_arguments();
    test_load_archive_metadata();
    test_load_tensor_payloads();
    test_shared_storage_payloads();
    test_invalid_tensor_payloads();
    test_constant_and_empty_payloads();
    test_missing_and_invalid_payloads();
    test_tensor_range_overflow();
    test_invalid_schema();
    test_argument_variants();
    test_multiple_models_are_rejected();

    if (test_failures != 0)
    {
        fprintf(stderr, "%d exported program test(s) failed\n", test_failures);
        return 1;
    }

    return 0;
}