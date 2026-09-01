// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_pt2.h"
#include "pt2_schema.h"
#include "aten_defaults_table.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

namespace pnnx {

// torch 的 tensor dtype 序列化枚举(pt2 权重 config / tensor_values 的 dtype
// 字段)→ pnnx type。实测(docs/11 §7):7 = float32,5 = int64。未知枚举返回 0。
static int pt2_dtype_enum_to_pnnx_type(long long dtype)
{
    switch (dtype)
    {
    case 7: return 1; // f32
    case 5: return 5; // i64
    default: return 0;
    }
}

static int pt2_dtype_to_pnnx_type(long long dtype)
{
    const int type = pt2_dtype_enum_to_pnnx_type(dtype);
    if (type == 0)
        fprintf(stderr, "load_pt2: unsupported weight dtype %lld\n", dtype);
    return type;
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

// "torch.ops.aten.conv2d.default" → "aten::conv2d.default"(保留 overload)
// map_pt2_target 剥 overload 是为了对齐 torchscript kind display;默认值静态表
// 按注册表全名(含 overload)组织,查表用本函数。
static std::string pt2_full_target_name(const std::string& target)
{
    const std::string prefix = "torch.ops.";
    if (target.compare(0, prefix.size(), prefix) != 0)
        return target;

    const std::string rest = target.substr(prefix.size()); // "aten.conv2d.default"

    const size_t dot1 = rest.find('.');
    if (dot1 == std::string::npos)
        return rest;

    return rest.substr(0, dot1) + "::" + rest.substr(dot1 + 1);
}

// 默认值静态表的编码值 → prim::Constant 的 value 参数。
// 返回 false 表示编码异常(生成器已按可编码性过滤,忠实起见不猜测)。
static bool default_value_to_parameter(int type, const char* value, Parameter& p)
{
    switch (type)
    {
    case PT2_D_NONE:
        p = Parameter();
        return true;
    case PT2_D_INT:
        p = Parameter((long long)strtoll(value, 0, 10));
        return true;
    case PT2_D_FLOAT:
        p = Parameter((float)strtod(value, 0));
        return true;
    case PT2_D_BOOL:
        p = Parameter(value[0] == '1');
        return true;
    case PT2_D_STRING:
        p = Parameter(std::string(value));
        return true;
    case PT2_D_INTS:
    {
        // 空列表 → Parameter()(type 0):ts 侧空列表实参物化为 None 常量
        // (如 max_pool2d stride=() 在 trace 图里是 value=None),下游转换器
        // 按 type 0 解释;pt2 补全必须对齐同一形态
        if (value[0] == '\0')
        {
            p = Parameter();
            return true;
        }

        std::vector<int> ai;
        const char* pch = value;
        while (*pch != '\0')
        {
            ai.push_back((int)strtoll(pch, 0, 10));
            pch = strchr(pch, ',');
            if (!pch)
                break;
            pch++;
        }
        p = Parameter(ai);
        return true;
    }
    case PT2_D_FLOATS:
    {
        if (value[0] == '\0')
        {
            p = Parameter();
            return true;
        }

        std::vector<float> af;
        const char* pch = value;
        while (*pch != '\0')
        {
            af.push_back((float)strtod(pch, 0));
            pch = strchr(pch, ',');
            if (!pch)
                break;
            pch++;
        }
        p = Parameter(af);
        return true;
    }
    case PT2_D_STRINGS:
    {
        if (value[0] == '\0')
        {
            p = Parameter();
            return true;
        }

        std::vector<std::string> as;
        const char* pch = value;
        while (*pch != '\0')
        {
            const char* comma = strchr(pch, ',');
            const size_t len = comma ? (size_t)(comma - pch) : strlen(pch);
            as.push_back(std::string(pch, len));
            if (!comma)
                break;
            pch = comma + 1;
        }
        p = Parameter(as);
        return true;
    }
    case PT2_D_DEVICE:
        // "cpu"/"cuda:0" 形态按 STRING 表达;"" 表示 None
        if (value[0] == '\0')
        {
            p = Parameter();
        }
        else
        {
            p = Parameter(std::string(value));
        }
        return true;
    default:
        return false;
    }
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

// 常量前置:builder 惰性创建标量 prim::Constant,在消费者之后追加。
// fuse_expression(pass_level3)从图尾向图头扫描,常量若先于其消费者被扫到
// 会先被包成 pnnx.Expression,消费者的算术链融合时无法内联成字面量,产出与
// torchscript 侧不同的常量 blob 形态(torchscript 侧常量总在消费者之前)。
// 将每个 prim::Constant 移到其(首个)消费者之前即满足该不变量;其余 op 保持
// 导出节点序(= 程序序),与 torchscript 侧一致。常量无输入,位置自由。
static void hoist_constants(Graph& pg)
{
    for (size_t i = 0; i < pg.ops.size(); i++)
    {
        Operator* op = pg.ops[i];
        if (op->type != "prim::Constant")
            continue;

        size_t consumer_pos = pg.ops.size();
        for (size_t j = 0; j < op->outputs.size(); j++)
        {
            const std::vector<Operator*>& consumers = op->outputs[j]->consumers;
            for (size_t k = 0; k < consumers.size(); k++)
            {
                size_t pos = std::find(pg.ops.begin(), pg.ops.end(), consumers[k]) - pg.ops.begin();
                if (pos < consumer_pos)
                    consumer_pos = pos;
            }
        }

        // 常量在消费者之前(或无消费者)则不动;无消费者的由 dead_code_elimination 清理
        if (consumer_pos >= pg.ops.size() || consumer_pos > i)
            continue;

        // 常量在消费者之后(builder 惰性创建的常态)→ 前移到消费者之前
        pg.ops.erase(pg.ops.begin() + i);
        pg.ops.insert(pg.ops.begin() + consumer_pos, op);
        i--; // 补偿 erase 的位置偏移,下一轮从原位继续
    }
}

// 切分族:torch.export 图内直接产出多输出(getitem 已被 exporter 折叠为
// as_tensors 列表),torchscript 侧是 1 输出 + prim::ListUnpack 形态,由
// fuse_op1ton_unpack(level3)折叠回多输出。转写为 ts 同构形态,使现有
// torch_unbind / torch_split / torch_chunk / torch_tensor_split 形态分支
// (1 输出 pattern)零改动匹配。
static bool pt2_target_unpackable(const std::string& op_type)
{
    return op_type == "aten::unbind" || op_type == "aten::split"
           || op_type == "aten::split_with_sizes" || op_type == "aten::chunk"
           || op_type == "aten::tensor_split";
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

            // 形状/dtype:文件内 tensor_values 是导出时的事实,优先采用
            // (CLI inputshape 与导出形状不一致时保证图自洽);CLI 仅作兜底
            std::map<std::string, Pt2TensorMeta>::const_iterator it = program.tensor_values.find(spec.graph_name);
            if (it != program.tensor_values.end())
            {
                r->type = pt2_dtype_enum_to_pnnx_type(it->second.dtype);
                for (size_t j = 0; j < it->second.sizes.size(); j++)
                    r->shape.push_back((int)it->second.sizes[j]);
            }

            if (input_index < (int)input_shapes.size())
            {
                if (r->type == 0)
                    r->type = pnnx_type_from_string(input_types[input_index]);

                if (r->shape.empty())
                {
                    for (size_t j = 0; j < input_shapes[input_index].size(); j++)
                        r->shape.push_back((int)input_shapes[input_index][j]);
                }
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

        // 默认值静态表:torch.export 会省略等于默认值的实参(cat 的 dim=0、
        // flatten 的 end_dim=-1 等),按 schema 形参序查表补全为完整形态,
        // 使 pt2 图与 torchscript 图同构,下游 pass_level2 形态分支零改动复用。
        // 表未收录的算子保持 torch.export 原样转写(缺参不补)。
        const std::string full_target = pt2_full_target_name(node.target);
        const Pt2DefaultsEntry* defaults = find_pt2_aten_defaults(full_target.c_str());

        // 待转写实参序列:有表 = schema 形参序(provided 按名归位,缺失留空待补);
        // provided 形参名与表对不上(schema 漂移等)= 整节点回退原样顺序(忠实性护栏)
        std::vector<const Pt2NodeInput*> ordered_inputs;
        if (defaults)
        {
            std::map<std::string, size_t> table_index;
            for (size_t j = 0; j < defaults->arg_count; j++)
                table_index[defaults->args[j].name] = j;

            bool table_matches = true;
            for (size_t j = 0; j < node.inputs.size(); j++)
            {
                if (table_index.find(node.inputs[j].name) == table_index.end())
                {
                    table_matches = false;
                    break;
                }
            }

            if (table_matches)
            {
                ordered_inputs.resize(defaults->arg_count, 0);
                for (size_t j = 0; j < node.inputs.size(); j++)
                {
                    ordered_inputs[table_index[node.inputs[j].name]] = &node.inputs[j];
                }
            }
        }

        if (ordered_inputs.empty())
        {
            if (defaults)
            {
                fprintf(stderr, "load_pt2: %s node %s: arg names mismatch defaults table, fallback to raw order\n",
                        full_target.c_str(), node.name.c_str());
            }

            for (size_t j = 0; j < node.inputs.size(); j++)
            {
                ordered_inputs.push_back(&node.inputs[j]);
            }
        }

        for (size_t j = 0; j < ordered_inputs.size(); j++)
        {
            const Pt2NodeInput* input = ordered_inputs[j];

            if (input == 0)
            {
                // 该形参被 torch.export 省略 → 查默认值补全(输出标记,便于排查)
                const Pt2ArgDefault& d = defaults->args[j];

                Parameter value;
                if (d.type == PT2_D_NO_DEFAULT || d.type == PT2_D_UNSUPPORTED
                    || !default_value_to_parameter(d.type, d.value, value))
                {
                    fprintf(stderr, "load_pt2: %s node %s: missing arg %s has no usable default, skipped\n",
                            full_target.c_str(), node.name.c_str(), d.name);
                    continue;
                }

                fprintf(stderr, "load_pt2: %s node %s: fill default %s=%s (from defaults table)\n",
                        full_target.c_str(), node.name.c_str(), d.name, d.value);

                Operator* op_const = pg.new_operator("prim::Constant",
                                                     "pnnx_" + std::to_string(pnnx_unknown_index++));
                op_const->params["value"] = value;

                Operand* r = pg.new_operand(node.name + "." + d.name);
                r->producer = op_const;
                op_const->outputs.push_back(r);

                r->consumers.push_back(op);
                op->inputs.push_back(r);
                continue;
            }

            const Pt2Argument& arg = input->arg;

            if (arg.type == Pt2Argument::TENSOR)
            {
                if (arg.tensor_names.size() != 1)
                {
                    fprintf(stderr, "load_pt2: bad tensor argument %s.%s\n", node.name.c_str(), input->name.c_str());
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

                Operand* r = pg.new_operand(node.name + "." + input->name);
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

            Operand* r = pg.new_operand(node.name + "." + input->name);
            r->producer = op_const;
            op_const->outputs.push_back(r);

            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }

        // 收集节点全部输出张量名(torch 2.13 的元组输出 = 单 spec 多 as_tensors)
        std::vector<std::string> out_tensor_names;
        for (size_t j = 0; j < node.outputs.size(); j++)
        {
            for (size_t k = 0; k < node.outputs[j].tensor_names.size(); k++)
            {
                out_tensor_names.push_back(node.outputs[j].tensor_names[k]);
            }
        }

        if (pt2_target_unpackable(op->type) && out_tensor_names.size() > 1)
        {
            // 切分族:1 输出(list)+ prim::ListUnpack,对齐 ts level1 形态
            Operand* list_out = pg.new_operand(node.name + ".out");
            list_out->producer = op;
            op->outputs.push_back(list_out);

            Operator* op_unpack = pg.new_operator("prim::ListUnpack",
                                                  "pnnx_" + std::to_string(pnnx_unknown_index++));

            list_out->consumers.push_back(op_unpack);
            op_unpack->inputs.push_back(list_out);

            for (size_t j = 0; j < out_tensor_names.size(); j++)
            {
                Operand* r = pg.new_operand(out_tensor_names[j]);
                r->producer = op_unpack;
                op_unpack->outputs.push_back(r);
            }
        }
        else
        {
            for (size_t j = 0; j < out_tensor_names.size(); j++)
            {
                Operand* r = pg.new_operand(out_tensor_names[j]);
                r->producer = op;
                op->outputs.push_back(r);
            }
        }
    }

    // 3.5 graph.tensor_values → operand 形状/dtype 补全
    //     torch.export 把 FakeTensor 元数据放 graph.tensor_values(含中间张量);
    //     pass_ncnn 的 shape 依赖转换器(torch_stack/adaptive_pool 等)需要它,
    //     CLI inputshape 只作用于 pnnx.Input,中间张量只能靠这张表。
    for (size_t i = 0; i < pg.operands.size(); i++)
    {
        Operand* r = pg.operands[i];

        std::map<std::string, Pt2TensorMeta>::const_iterator it = program.tensor_values.find(r->name);
        if (it == program.tensor_values.end())
            continue;

        if (r->type == 0)
            r->type = pt2_dtype_enum_to_pnnx_type(it->second.dtype);

        if (r->shape.empty())
        {
            for (size_t j = 0; j < it->second.sizes.size(); j++)
                r->shape.push_back((int)it->second.sizes[j]);
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

    // 5. 常量前置:满足 fuse_expression 反向扫描的内联不变量(见函数注释)
    hoist_constants(pg);

    return 0;
}

} // namespace pnnx
