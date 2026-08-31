// Tencent 2026
// SPDX-License-Identifier: BSD-3-Clause

// pt2_schema 解析 dump(独立 harness,N1 验收工具)。
// 编译(在 tests/ 目录下):
//     g++ -O2 -std=c++11 -Wall -Wextra -I../src test_pt2_schema.cpp ../src/pt2_schema.cpp ../src/storezip.cpp -o test_pt2_schema
// 用法:
//     ./test_pt2_schema model.pt2          人类可读 dump
//     ./test_pt2_schema -c model.pt2       规范化输出(与 python 侧逐行对拍用)
//     ./test_pt2_schema -q model.pt2       仅退出码(0 = 解析成功)
//
// 对拍:pt2-dump/dump_canonical.py 用同一模型生成规范化文本,两者 diff 必须为空。

#include "pt2_schema.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace pnnx;

// 规范化文本:每节点一行 "target|name|in:参数名=形态:引用名,|out:名,",
// 形态记号 t=张量引用 ts=张量列表 i=整 iv=整数表 f=浮点 fv=浮点表 b=布尔
// bv=布尔表 s=字符串 sv=字符串表 st=dtype dev=设备 mf=内存格式 n=none
// 参数名前缀 k$ 表示 keyword 实参。

static const char* arg_shape(const Pt2Argument& a)
{
    switch (a.type)
    {
    case Pt2Argument::TENSOR: return "t";
    case Pt2Argument::TENSORS: return "ts";
    case Pt2Argument::INT: return "i";
    case Pt2Argument::INTS: return "iv";
    case Pt2Argument::FLOAT: return "f";
    case Pt2Argument::FLOATS: return "fv";
    case Pt2Argument::BOOL: return "b";
    case Pt2Argument::BOOLS: return "bv";
    case Pt2Argument::STRING: return "s";
    case Pt2Argument::STRINGS: return "sv";
    case Pt2Argument::SCALAR_TYPE: return "st";
    case Pt2Argument::DEVICE: return "dev";
    case Pt2Argument::MEMORY_FORMAT: return "mf";
    case Pt2Argument::NONE:
    default: return "n";
    }
}

static void print_argument_list(std::string& line, const std::vector<Pt2NodeInput>& inputs)
{
    for (size_t i = 0; i < inputs.size(); i++)
    {
        const Pt2Argument& a = inputs[i].arg;
        char buf[512];
        snprintf(buf, sizeof(buf), "%s%s=%s", a.is_kwarg ? "k$" : "", a.name.c_str(), arg_shape(a));
        line += buf;
        if (a.type == Pt2Argument::TENSOR || a.type == Pt2Argument::TENSORS)
        {
            line += ":";
            for (size_t j = 0; j < a.tensor_names.size(); j++)
            {
                line += a.tensor_names[j];
                if (j + 1 < a.tensor_names.size())
                    line += ",";
            }
        }
        if (i + 1 < inputs.size())
            line += ",";
    }
}

static void print_node_outputs(std::string& line, const std::vector<Pt2NodeOutput>& outputs)
{
    for (size_t i = 0; i < outputs.size(); i++)
    {
        for (size_t j = 0; j < outputs[i].tensor_names.size(); j++)
        {
            line += outputs[i].tensor_names[j];
            if (j + 1 < outputs[i].tensor_names.size() || i + 1 < outputs.size())
                line += ",";
        }
    }
}

static void dump_canonical(const Pt2Program& program)
{
    // header
    printf("schema_version=%lld.%lld\ntorch_version=%s\nroot=%s\n", program.schema_version_major,
           program.schema_version_minor, program.torch_version.c_str(), program.archive_root.c_str());

    printf("opset:");
    for (std::map<std::string, std::string>::const_iterator it = program.opset_version.begin();
         it != program.opset_version.end(); ++it)
    {
        printf("%s=%s,", it->first.c_str(), it->second.c_str());
    }
    printf("\n");

    // input specs(按 kind 排序键:user_input=0 parameter=1 buffer=2 tensor_constant=3)
    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const Pt2InputSpec& s = program.input_specs[i];
        static const char* kind_names[] = {"user_input", "parameter", "buffer", "tensor_constant"};
        printf("in_spec:%s:%s->%s\n", kind_names[s.kind], s.graph_name.c_str(), s.state_dict_name.c_str());
    }

    for (size_t i = 0; i < program.output_specs.size(); i++)
    {
        printf("out_spec:%s\n", program.output_specs[i].graph_name.c_str());
    }

    // nodes
    for (size_t i = 0; i < program.nodes.size(); i++)
    {
        const Pt2Node& node = program.nodes[i];
        std::string line = "node:";
        line += node.target;
        line += "|";
        print_argument_list(line, node.inputs);
        line += "|out:";
        print_node_outputs(line, node.outputs);
        printf("%s\n", line.c_str());
    }

    // weights / constants
    for (size_t i = 0; i < program.weights.size(); i++)
    {
        const Pt2WeightEntry& w = program.weights[i];
        printf("weight:%s->%s:dtype=%lld:sz=", w.state_dict_name.c_str(), w.path_name.c_str(), w.dtype);
        for (size_t j = 0; j < w.sizes.size(); j++)
            printf("%s%lld", j ? "x" : "", w.sizes[j]);
        printf("\n");
    }
    for (size_t i = 0; i < program.constants.size(); i++)
    {
        const Pt2WeightEntry& w = program.constants[i];
        printf("constant:%s->%s:dtype=%lld:sz=", w.state_dict_name.c_str(), w.path_name.c_str(), w.dtype);
        for (size_t j = 0; j < w.sizes.size(); j++)
            printf("%s%lld", j ? "x" : "", w.sizes[j]);
        printf("\n");
    }
}

static void dump_human(const Pt2Program& program)
{
    printf("=== pt2 schema: archive_root=%s\n", program.archive_root.c_str());
    printf("    schema_version=%lld.%lld  torch_version=%s\n", program.schema_version_major,
           program.schema_version_minor, program.torch_version.c_str());
    printf("    opset_version:");
    for (std::map<std::string, std::string>::const_iterator it = program.opset_version.begin();
         it != program.opset_version.end(); ++it)
    {
        printf(" %s=%s", it->first.c_str(), it->second.c_str());
    }
    printf("\n");

    printf("=== input_specs (%zu)\n", program.input_specs.size());
    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const Pt2InputSpec& s = program.input_specs[i];
        static const char* kind_names[] = {"user_input", "parameter", "buffer", "tensor_constant"};
        if (s.state_dict_name.empty())
            printf("  [%zu] %-16s %s\n", i, kind_names[s.kind], s.graph_name.c_str());
        else
            printf("  [%zu] %-16s %s -> %s%s\n", i, kind_names[s.kind], s.graph_name.c_str(),
                   s.state_dict_name.c_str(), s.persistent ? " (persistent)" : "");
    }

    printf("=== output_specs (%zu)\n", program.output_specs.size());
    for (size_t i = 0; i < program.output_specs.size(); i++)
    {
        printf("  [%zu] user_output %s\n", i, program.output_specs[i].graph_name.c_str());
    }

    printf("=== nodes (%zu)\n", program.nodes.size());
    for (size_t i = 0; i < program.nodes.size(); i++)
    {
        const Pt2Node& node = program.nodes[i];
        printf("  [%zu] %s  name=%s\n", i, node.target.c_str(), node.name.c_str());
        if (!node.nn_module_stack.empty())
            printf("      nn_module_stack: %s\n", node.nn_module_stack.c_str());
        if (!node.torch_fn.empty())
            printf("      torch_fn: %s\n", node.torch_fn.c_str());
        for (size_t j = 0; j < node.inputs.size(); j++)
        {
            const Pt2Argument& a = node.inputs[j].arg;
            printf("      %s%s = ", a.is_kwarg ? "kw " : "   ", node.inputs[j].name.c_str());
            switch (a.type)
            {
            case Pt2Argument::TENSOR:
                printf("tensor(%s)", a.tensor_names[0].c_str());
                break;
            case Pt2Argument::TENSORS:
                printf("tensors(");
                for (size_t k = 0; k < a.tensor_names.size(); k++)
                    printf("%s%s", k ? ", " : "", a.tensor_names[k].c_str());
                printf(")");
                break;
            case Pt2Argument::INT:
                printf("int %lld", a.int_value);
                break;
            case Pt2Argument::INTS:
                printf("ints [");
                for (size_t k = 0; k < a.int_values.size(); k++)
                    printf("%s%lld", k ? ", " : "", a.int_values[k]);
                printf("]");
                break;
            case Pt2Argument::FLOAT:
                printf("float %g", a.float_value);
                break;
            case Pt2Argument::FLOATS:
                printf("floats [");
                for (size_t k = 0; k < a.float_values.size(); k++)
                    printf("%s%g", k ? ", " : "", a.float_values[k]);
                printf("]");
                break;
            case Pt2Argument::BOOL:
                printf("bool %s", a.bool_value ? "true" : "false");
                break;
            case Pt2Argument::BOOLS:
                printf("bools [");
                for (size_t k = 0; k < a.bool_values.size(); k++)
                    printf("%s%d", k ? ", " : "", a.bool_values[k] ? 1 : 0);
                printf("]");
                break;
            case Pt2Argument::STRING:
                printf("string \"%s\"", a.string_value.c_str());
                break;
            case Pt2Argument::STRINGS:
                printf("strings [");
                for (size_t k = 0; k < a.string_values.size(); k++)
                    printf("%s\"%s\"", k ? ", " : "", a.string_values[k].c_str());
                printf("]");
                break;
            case Pt2Argument::SCALAR_TYPE:
                printf("scalar_type %lld", a.int_value);
                break;
            case Pt2Argument::DEVICE:
                printf("device {type=%s, index=%lld}", a.device_type.c_str(), a.device_index);
                break;
            case Pt2Argument::MEMORY_FORMAT:
                printf("memory_format %lld", a.int_value);
                break;
            case Pt2Argument::NONE:
            default:
                printf("none");
                break;
            }
            printf("\n");
        }
        for (size_t j = 0; j < node.outputs.size(); j++)
        {
            printf("      out ");
            for (size_t k = 0; k < node.outputs[j].tensor_names.size(); k++)
                printf("%s%s", k ? ", " : "", node.outputs[j].tensor_names[k].c_str());
            printf("\n");
        }
    }

    printf("=== weights (%zu)\n", program.weights.size());
    for (size_t i = 0; i < program.weights.size(); i++)
    {
        const Pt2WeightEntry& w = program.weights[i];
        printf("  [%zu] %s -> %s  dtype=%lld is_param=%d use_pickle=%d sizes=", i,
               w.state_dict_name.c_str(), w.path_name.c_str(), w.dtype, w.is_param ? 1 : 0,
               w.use_pickle ? 1 : 0);
        for (size_t j = 0; j < w.sizes.size(); j++)
            printf("%s%lld", j ? "x" : "", w.sizes[j]);
        printf("\n");
    }

    printf("=== constants (%zu)\n", program.constants.size());
    for (size_t i = 0; i < program.constants.size(); i++)
    {
        const Pt2WeightEntry& w = program.constants[i];
        printf("  [%zu] %s -> %s  dtype=%lld sizes=", i, w.state_dict_name.c_str(),
               w.path_name.c_str(), w.dtype);
        for (size_t j = 0; j < w.sizes.size(); j++)
            printf("%s%lld", j ? "x" : "", w.sizes[j]);
        printf("\n");
    }
}

int main(int argc, char** argv)
{
    const char* mode = "human";
    const char* path = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-c") == 0)
            mode = "canonical";
        else if (strcmp(argv[i], "-q") == 0)
            mode = "quiet";
        else
            path = argv[i];
    }

    if (!path)
    {
        fprintf(stderr, "Usage: test_pt2_schema [-c|-q] model.pt2\n");
        return 1;
    }

    Pt2Program program;
    if (load_pt2_schema(path, program) != 0)
    {
        fprintf(stderr, "load_pt2_schema failed: %s\n", path);
        return 1;
    }

    if (strcmp(mode, "canonical") == 0)
        dump_canonical(program);
    else if (strcmp(mode, "human") == 0)
        dump_human(program);
    // quiet: 仅验证解析成功,无输出

    return 0;
}
