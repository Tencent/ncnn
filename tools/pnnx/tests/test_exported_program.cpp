// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>

#include <string>

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

static void test_parse_exported_program()
{
    pnnx::pt2::ExportedProgram program;
    std::string error;
    expect_true(pnnx::pt2::parse_exported_program(exported_program_json, program, error), error.c_str());
    expect_true(program.schema_version.major == 8 && program.schema_version.minor == 20, "schema version");
    expect_true(program.graph.nodes.size() == 1 && program.graph.nodes[0].name == "linear", "graph node");
    expect_true(program.graph.nodes[0].inputs.size() == 2 && program.graph.nodes[0].inputs[0].kind == pnnx::pt2::NamedArgument::Positional, "named positional arguments");
    expect_true(program.signature.inputs.size() == 2 && program.signature.inputs[0].target == "linear.weight", "graph signature");
    expect_true(program.graph.tensor_values["x"].sizes[0].has_hint && program.graph.tensor_values["x"].sizes[0].hint == 2, "symbolic shape hint");
    expect_true(program.range_constraints["s0"].has_min && !program.range_constraints["s0"].has_max, "unbounded range constraint");
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
    expect_true(pnnx::pt2::load_exported_program_archive_metadata(path, archive, error), error.c_str());
    expect_true(archive.model_name == "model", "archive model name");
    expect_true(archive.program.graph.nodes[0].target == "torch.ops.aten.linear.default", "archive graph target");
    remove(path);
}

static void write_payload_config(pnnx::StoreZipWriter& writer, const char* name, const char* path, bool is_parameter, const char* sizes, const char* strides, int storage_offset = 0, bool use_pickle = false)
{
    const std::string config = std::string("{\"config\":{\"tensor\":{\"path_name\":\"") + path
                               + "\",\"is_param\":" + (is_parameter ? "true" : "false")
                               + ",\"use_pickle\":" + (use_pickle ? "true" : "false")
                               + ",\"tensor_meta\":{\"dtype\":7,\"sizes\":" + sizes
                               + ",\"requires_grad\":false,\"device\":{\"type\":\"cpu\"},\"strides\":" + strides
                               + ",\"storage_offset\":{\"as_int\":" + std::to_string(storage_offset) + "},\"layout\":7}}}}";
    writer.write_file(name, config.data(), config.size());
}

static void write_empty_payload_config(pnnx::StoreZipWriter& writer, const char* name)
{
    const char* config = "{\"config\":{}}";
    writer.write_file(name, config, std::string(config).size());
}

static void write_payload_archive(const char* path, const char* payload_path, const char* sizes, const char* strides, int storage_offset, const std::vector<char>& storage, bool use_pickle = false)
{
    pnnx::StoreZipWriter writer;
    writer.open(path);
    writer.write_file("package/archive_format", "pt2", 3);
    writer.write_file("package/archive_version", "0", 1);
    writer.write_file("package/models/model.json", exported_program_json, std::string(exported_program_json).size());
    write_payload_config(writer, "package/data/weights/model_weights_config.json", payload_path, true, sizes, strides, storage_offset, use_pickle);
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
    expect_true(pnnx::pt2::load_exported_program_archive(path, archive, error), error.c_str());
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
    expect_true(pnnx::pt2::load_exported_program_archive(path, archive, error), error.c_str());
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
    expect_true(pnnx::pt2::load_exported_program_archive(path, archive, error), error.c_str());
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
    const char* document = R"json({"graph_module":{"graph":{"inputs":[],"outputs":[],"nodes":[{"target":"torch.ops.aten.index.Tensor","inputs":[{"name":"indices","arg":{"as_optional_tensors":[{"as_none":true},{"as_tensor":{"name":"index"}}]},"kind":1}],"outputs":[],"metadata":{}}],"tensor_values":{},"sym_int_values":{}},"signature":{"input_specs":[],"output_specs":[]}},"opset_version":{"aten":1},"range_constraints":{},"schema_version":{"major":8,"minor":20}})json";
    pnnx::pt2::ExportedProgram program;
    std::string error;
    expect_true(pnnx::pt2::parse_exported_program(document, program, error), error.c_str());
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

int main()
{
    test_defaults();
    test_minimal_archive();
    test_parse_exported_program();
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