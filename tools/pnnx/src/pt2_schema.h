// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PT2_SCHEMA_H
#define PNNX_PT2_SCHEMA_H

#include "storezip.h"

#include <stdint.h>
#include <map>
#include <string>
#include <vector>

// torch.export ExportedProgram (.pt2) 的序列化 schema。
//
// 事实来源:docs/11-pt2-schema-reference.md + pt2-dump/survey_all.py 对 237 个
// 真实 .pt2(torch 2.13.0 导出)的全量实测(2026-08-31)。不同 torch 小版本
// 字段可能有差异,解析按 schema_version 容错。
//
// 容器布局(zip 内,所有条目带以模型名命名的根目录前缀):
//     <root>/models/model.json                    图 + signature
//     <root>/data/weights/weight_N                裸权重二进制
//     <root>/data/weights/model_weights_config.json  state_dict名 → weight_N 映射
//     <root>/data/constants/model_constants_config.json
//
// 设计原则:忠实转写。本层保持 JSON 原始形态(参数三态:张量引用/字面量/None),
// 不做任何语义归一;归一化属于 pass_level2 的 PT2 形态分支。

namespace pnnx {

// 节点参数。对应 node.inputs[].arg,按 as_* 变体区分。
struct Pt2Argument
{
    enum ArgType
    {
        NONE,
        TENSOR,       // as_tensor       {name}                 单个张量引用
        TENSORS,      // as_tensors      [{name}, ...]          张量引用列表
        INT,          // as_int          整数字面量
        INTS,         // as_ints         整数列表字面量
        FLOAT,        // as_float        浮点字面量
        FLOATS,       // as_floats       浮点列表字面量
        BOOL,         // as_bool         布尔字面量
        BOOLS,        // as_bools        布尔列表字面量
        STRING,       // as_string       字符串字面量
        STRINGS,      // as_strings      字符串列表字面量
        SCALAR_TYPE,  // as_scalar_type  dtype 枚举值(int)
        DEVICE,       // as_device       {type, index}
        MEMORY_FORMAT // as_memory_format 枚举值(int)
    };

    ArgType type;
    bool is_kwarg;    // node.inputs[].kind: 1 = positional, 2 = keyword
    std::string name; // 形参名(self / weight / stride / ...)

    std::vector<std::string> tensor_names;  // TENSOR(恰 1 个)/ TENSORS(N 个)
    long long int_value;                    // INT / SCALAR_TYPE / MEMORY_FORMAT
    std::vector<long long> int_values;      // INTS
    double float_value;                     // FLOAT
    std::vector<double> float_values;       // FLOATS
    bool bool_value;                        // BOOL
    std::vector<bool> bool_values;          // BOOLS
    std::string string_value;               // STRING
    std::vector<std::string> string_values; // STRINGS
    std::string device_type;                // DEVICE
    long long device_index;                 // DEVICE,-1 表示 null

    Pt2Argument()
        : type(NONE), is_kwarg(false), int_value(0), float_value(0), bool_value(false), device_index(-1)
    {
    }
};

struct Pt2NodeInput
{
    std::string name; // 形参名
    Pt2Argument arg;
};

struct Pt2NodeOutput
{
    // as_tensor(恰 1 个)或 as_tensors(N 个);元素即输出张量名(ncnn blob 名)
    std::vector<std::string> tensor_names;
};

struct Pt2Node
{
    std::string name;   // 节点名(flatten / relu / ...)
    std::string target; // "torch.ops.aten.<op>.<overload>"
    std::vector<Pt2NodeInput> inputs;
    std::vector<Pt2NodeOutput> outputs;

    // node.metadata(诊断与 pass_level2 形态判定用)
    std::string nn_module_stack; // "L__self__,,__main__.M" 形态的紧凑编码
    std::string torch_fn;
    std::string stack_trace;
};

struct Pt2InputSpec
{
    enum SpecKind
    {
        USER_INPUT,
        PARAMETER,
        BUFFER,
        TENSOR_CONSTANT
    };

    SpecKind kind;
    std::string graph_name;      // arg.name(图内张量名,如 p_conv_weight / x / c_m3)
    std::string state_dict_name; // parameter_name / buffer_name / tensor_constant_name
    bool persistent;             // buffer 专属字段

    Pt2InputSpec()
        : kind(USER_INPUT), persistent(false)
    {
    }
};

struct Pt2OutputSpec
{
    std::string graph_name; // user_output 的张量名
};

// model_weights_config.json / model_constants_config.json 的一个条目
struct Pt2WeightEntry
{
    std::string state_dict_name;
    std::string path_name; // "weight_N"
    bool is_param;
    bool use_pickle;
    long long dtype; // tensor_meta.dtype,7 = float32,5 = int64(实测)
    std::vector<long long> sizes;
    std::vector<long long> strides;
    long long storage_offset;

    Pt2WeightEntry()
        : is_param(false), use_pickle(false), dtype(0), storage_offset(0)
    {
    }
};

// 一个 .pt2 文件的完整 schema(model.json + weights/constants config)
struct Pt2Program
{
    // header
    // schema_version 实测为结构化对象 {"major": 8, "minor": 20};-1 表示缺失
    long long schema_version_major;
    long long schema_version_minor;
    std::string torch_version;                        // 实测 "2.13.0"
    std::map<std::string, std::string> opset_version; // 实测 {"aten": "10"}
    std::string archive_root;                         // zip 内根目录前缀,含尾部斜杠(如 "mini/")

    Pt2Program()
        : schema_version_major(-1), schema_version_minor(-1)
    {
    }

    std::vector<Pt2Node> nodes;
    std::vector<Pt2InputSpec> input_specs;
    std::vector<Pt2OutputSpec> output_specs;

    std::vector<Pt2WeightEntry> weights;   // model_weights_config.json
    std::vector<Pt2WeightEntry> constants; // model_constants_config.json(实测常为空)

    // 权重二进制在 zip 内的完整条目路径(<root>data/weights/weight_N)
    std::string weight_entry_path(const std::string& path_name) const;

    const Pt2WeightEntry* find_weight(const std::string& state_dict_name) const;
    const Pt2WeightEntry* find_constant(const std::string& state_dict_name) const;
};

// 解析 .pt2 的 schema 层(header / nodes / signature / weights_config 四块),
// 不读权重二进制字节。失败返回非 0 并输出错误信息到 stderr。
int load_pt2_schema(const std::string& ptpath, Pt2Program& program);

// 在 zip 条目列表中定位 <root>/models/model.json,返回该条目名;找不到返回空串。
// 这是 .pt2 的判定特征(model_file_maybe_pt2 与 load_pt2_schema 共用)。
std::string find_pt2_model_json_entry(const std::vector<std::string>& entry_names);

} // namespace pnnx

#endif // PNNX_PT2_SCHEMA_H
