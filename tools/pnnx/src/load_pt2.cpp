// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_pt2.h"
#include "pt2_schema.h"

#include <stdio.h>
#include <string.h>

namespace pnnx {

// torch 的 tensor dtype 序列化枚举(pt2 权重 config 的 dtype 字段)→ pnnx type
// 实测(docs/11 §7):7 = float32,5 = int64
static int pt2_dtype_to_pnnx_type(long long dtype)
{
    switch (dtype)
    {
    case 7: return 1; // f32
    case 5: return 5; // i64
    default:
        fprintf(stderr, "load_pt2: unsupported weight dtype %lld\n", dtype);
        return 0;
    }
}

static int pnnx_type_from_string(const std::string& t)
{
    if (t == "f32") return 1;
    if (t == "f64") return 2;
    if (t == "f16") return 3;
    if (t == "i32") return 4;
    if (t == "i64") return 5;
    if (t == "i16") return 6;
    if (t == "i8") return 7;
    if (t == "u8") return 8;
    if (t == "bool") return 9;
    return 0;
}

// "torch.ops.aten.conv2d.default" → "aten::conv2d"
// torchscript 的 kind display 不含 overload,pt2 target 必须剥掉后缀才能与
// 现有 pass_level2 形态分支(pnnx.Input 匹配)对齐
static std::string map_pt2_target(const std::string& target)
{
    const std::string prefix = "torch.ops.";
    if (target.compare(0, prefix.size(), prefix) != 0)
        return target;

    std::string rest = target.substr(prefix.size()); // "aten.conv2d.default"

    const size_t dot1 = rest.find('.');
    const size_t dot2 = rest.find('.', dot1 + 1);
    if (dot1 == std::string::npos)
        return rest;

    if (dot2 == std::string::npos)
        return rest.substr(0, dot1) + "::" + rest.substr(dot1 + 1);

    return rest.substr(0, dot1) + "::" + rest.substr(dot1 + 1, dot2 - dot1 - 1);
}

// 把 prim::Constant 的 value 参数从 Pt2Argument 转出来。
// 返回 false 表示该形态暂不支持(builder 忠实转写,不做语义猜测)。
static bool argument_to_constant(const Pt2Argument& a, Parameter& value)
{
    switch (a.type)
    {
    case Pt2Argument::NONE:
        value = Parameter();
        return true;
    case Pt2Argument::INT:
    case Pt2Argument::SCALAR_TYPE:
        value = Parameter((long long)a.int_value);
        return true;
    case Pt2Argument::INTS:
    {
        std::vector<int> ai;
        for (size_t k = 0; k < a.int_values.size(); k++)
            ai.push_back((int)a.int_values[k]);
        value = Parameter(ai);
        return true;
    }
    case Pt2Argument::FLOAT:
        value = Parameter((float)a.float_value);
        return true;
    case Pt2Argument::FLOATS:
        value = Parameter(a.float_values);
        return true;
    case Pt2Argument::BOOL:
        value = Parameter(a.bool_value);
        return true;
    case Pt2Argument::STRING:
        value = Parameter(a.string_value);
        return true;
    default:
        fprintf(stderr, "load_pt2: unsupported constant argument %s (%d)\n", a.name.c_str(), a.type);
        return false;
    }
}

// 从 zip 读取权重/常量二进制,构造 pnnx Attribute(裸字节,无 pickle)
static int load_weight_attribute(const Pt2Program& program, const Pt2WeightEntry& entry, bool is_constant, Attribute& attr)
{
    if (entry.use_pickle)
    {
        fprintf(stderr, "load_pt2: use_pickle weights not supported yet (%s)\n", entry.state_dict_name.c_str());
        return -1;
    }

    const std::string entry_path = is_constant ? program.constant_entry_path(entry.path_name)
                                               : program.weight_entry_path(entry.path_name);

    StoreZipReader zip;
    if (zip.open(program.zippath) != 0)
    {
        fprintf(stderr, "load_pt2: reopen zip failed %s\n", program.zippath.c_str());
        return -1;
    }

    const uint64_t raw_size = zip.get_file_size(entry_path);
    std::vector<char> raw((size_t)raw_size);
    int ret = zip.read_file(entry_path, raw.data());
    zip.close();
    if (ret != 0)
    {
        fprintf(stderr, "load_pt2: read weight failed %s\n", entry_path.c_str());
        return -1;
    }

    attr.type = pt2_dtype_to_pnnx_type(entry.dtype);
    if (attr.type == 0)
        return -1;

    const int elemsize = (attr.type == 1) ? 4 : ((attr.type == 5) ? 8 : 0);
    if (elemsize == 0)
    {
        fprintf(stderr, "load_pt2: unsupported attribute type %d\n", attr.type);
        return -1;
    }

    for (size_t i = 0; i < entry.sizes.size(); i++)
        attr.shape.push_back((int)entry.sizes[i]);

    size_t elem_count = 1;
    for (size_t i = 0; i < attr.shape.size(); i++)
        elem_count *= (size_t)attr.shape[i];

    if (raw.size() != elem_count * elemsize)
    {
        fprintf(stderr, "load_pt2: weight size mismatch %s: expect %zu got %llu\n", entry_path.c_str(),
                elem_count * elemsize, (unsigned long long)raw.size());
        return -1;
    }

    attr.data = raw;
    return 0;
}

int load_pt2(const std::string& ptpath, Graph& pg,
             const std::vector<std::vector<int64_t> >& input_shapes,
             const std::vector<std::string>& input_types)
{
    Pt2Program program;
    program.zippath = ptpath;

    int ret = load_pt2_schema(ptpath, program);
    if (ret != 0)
        return ret;

    fprintf(stderr, "load_pt2: schema_version=%lld.%lld torch_version=%s nodes=%zu params=%zu\n",
            program.schema_version_major, program.schema_version_minor, program.torch_version.c_str(),
            program.nodes.size(), program.weights.size());

    int pnnx_unknown_index = 0;

    // 1. user_input → pnnx.Input(与 ts level1 相同:pnnx_input_%d 序号)
    {
        int input_index = 0;
        for (size_t i = 0; i < program.input_specs.size(); i++)
        {
            const Pt2InputSpec& spec = program.input_specs[i];
            if (spec.kind != Pt2InputSpec::USER_INPUT)
                continue;

            char name[32];
            snprintf(name, sizeof(name), "pnnx_input_%d", input_index);

            Operator* op = pg.new_operator("pnnx.Input", name);
            Operand* r = pg.new_operand(spec.graph_name);
            r->producer = op;
            op->outputs.push_back(r);

            if (input_index < (int)input_shapes.size())
            {
                r->type = pnnx_type_from_string(input_types[input_index]);
                for (size_t j = 0; j < input_shapes[input_index].size(); j++)
                    r->shape.push_back((int)input_shapes[input_index][j]);
            }

            input_index++;
        }
    }

    // 2. parameter / buffer / tensor_constant → pnnx.Attribute
    //    (op->name = state_dict 名,与 ts level1 的 GetAttr wrapped_name 对齐)
    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const Pt2InputSpec& spec = program.input_specs[i];
        if (spec.kind == Pt2InputSpec::USER_INPUT)
            continue;

        const Pt2WeightEntry* entry = 0;
        if (spec.kind == Pt2InputSpec::TENSOR_CONSTANT)
            entry = program.find_constant(spec.state_dict_name);
        else
            entry = program.find_weight(spec.state_dict_name);

        if (!entry)
        {
            fprintf(stderr, "load_pt2: weight entry not found for %s (%s)\n", spec.graph_name.c_str(),
                    spec.state_dict_name.c_str());
            return -1;
        }

        Attribute attr;
        if (load_weight_attribute(program, *entry, spec.kind == Pt2InputSpec::TENSOR_CONSTANT, attr) != 0)
            return -1;

        Operator* op = pg.new_operator("pnnx.Attribute", spec.state_dict_name);
        op->attrs["data"] = attr;

        Operand* r = pg.new_operand(spec.graph_name);
        r->producer = op;
        op->outputs.push_back(r);
        r->type = attr.type;
        r->shape = attr.shape;
    }

    // 3. 图节点 → aten 原名 Operator,标量参数 operand 化(忠实转写,零归一化)
    for (size_t i = 0; i < program.nodes.size(); i++)
    {
        const Pt2Node& node = program.nodes[i];

        Operator* op = pg.new_operator(map_pt2_target(node.target), "pnnx_" + std::to_string(pnnx_unknown_index++));

        for (size_t j = 0; j < node.inputs.size(); j++)
        {
            const Pt2NodeInput& input = node.inputs[j];
            const Pt2Argument& arg = input.arg;

            if (arg.type == Pt2Argument::TENSOR)
            {
                if (arg.tensor_names.size() != 1)
                {
                    fprintf(stderr, "load_pt2: bad tensor argument %s.%s\n", node.name.c_str(), input.name.c_str());
                    return -1;
                }

                Operand* r = pg.get_operand(arg.tensor_names[0]);
                if (!r)
                {
                    fprintf(stderr, "load_pt2: operand not found %s (node %s)\n", arg.tensor_names[0].c_str(),
                            node.name.c_str());
                    return -1;
                }

                r->consumers.push_back(op);
                op->inputs.push_back(r);
                continue;
            }

            if (arg.type == Pt2Argument::TENSORS)
            {
                // ts 形态对齐:张量列表经 prim::ListConstruct 折成单个 list operand
                Operator* op_list = pg.new_operator("prim::ListConstruct",
                                                    "pnnx_" + std::to_string(pnnx_unknown_index++));

                for (size_t k = 0; k < arg.tensor_names.size(); k++)
                {
                    Operand* r = pg.get_operand(arg.tensor_names[k]);
                    if (!r)
                    {
                        fprintf(stderr, "load_pt2: operand not found %s (node %s)\n", arg.tensor_names[k].c_str(),
                                node.name.c_str());
                        return -1;
                    }
                    r->consumers.push_back(op_list);
                    op_list->inputs.push_back(r);
                }

                Operand* r = pg.new_operand(node.name + "." + input.name);
                r->producer = op_list;
                op_list->outputs.push_back(r);

                r->consumers.push_back(op);
                op->inputs.push_back(r);
                continue;
            }

            // 标量/None 字面量 → prim::Constant operand
            Parameter value;
            if (!argument_to_constant(arg, value))
                return -1;

            Operator* op_const = pg.new_operator("prim::Constant",
                                                 "pnnx_" + std::to_string(pnnx_unknown_index++));
            op_const->params["value"] = value;

            Operand* r = pg.new_operand(node.name + "." + input.name);
            r->producer = op_const;
            op_const->outputs.push_back(r);

            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }

        for (size_t j = 0; j < node.outputs.size(); j++)
        {
            for (size_t k = 0; k < node.outputs[j].tensor_names.size(); k++)
            {
                Operand* r = pg.new_operand(node.outputs[j].tensor_names[k]);
                r->producer = op;
                op->outputs.push_back(r);
                // 中间张量的 dtype/形状 pt2 JSON 不携带,留空由下游 pass 处理
            }
        }
    }

    // 4. user_output → pnnx.Output
    for (size_t i = 0; i < program.output_specs.size(); i++)
    {
        char name[32];
        snprintf(name, sizeof(name), "pnnx_output_%d", (int)i);

        Operator* op = pg.new_operator("pnnx.Output", name);

        Operand* r = pg.get_operand(program.output_specs[i].graph_name);
        if (!r)
        {
            fprintf(stderr, "load_pt2: output operand not found %s\n", program.output_specs[i].graph_name.c_str());
            return -1;
        }

        r->consumers.push_back(op);
        op->inputs.push_back(r);
    }

    return 0;
}

} // namespace pnnx
