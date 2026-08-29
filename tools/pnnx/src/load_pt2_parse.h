// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_LOAD_PT2_PARSE_H
#define PNNX_LOAD_PT2_PARSE_H

// 方案2 中间格式：Python 预处理已把 .pt2 转成"规范 pnnx IR"JSON，本结构是其 C++ 映射。
// 节点已是规范 pnnx 算子（pnnx_type + 已命名已解析 params + 内联 attrs + 解析后 output_shapes），
// C++ emit 只需纯建图，不再做 aten→pnnx 映射 / 形状推理 / 参数补全。
//
// 仅用 STL，不依赖 pnnx::Graph / ir.h，可独立单测。

#include <map>
#include <string>
#include <vector>

namespace pnnx {

// 规范参数值（对应 pnnx::Parameter 的几种变体）
struct Pt2Param
{
    int type = 0; // 0=null 1=bool 2=int 3=float 4=string 5=ints 6=floats
    bool b = false;
    long long i = 0;
    double f = 0.0;
    std::string s;
    std::vector<long long> ii;
    std::vector<double> ff;
};

// 内联权重属性（module 形态算子的 @weight/@bias 等）
struct Pt2Attr
{
    std::vector<int> shape;
    int dtype = 7; // 7=float32
    std::vector<float> data; // 已从 zip data/N 读出的 float32 字节
};

struct Pt2IO
{
    std::string name;
    std::vector<int> shape;
    int dtype = 7;
};

struct Pt2Node
{
    std::string name;       // op 名
    std::string pnnx_type;  // 规范 pnnx 算子类型，如 nn.Conv2d / Tensor.reshape / torch.cat
    std::vector<std::string> inputs;           // 输入 operand 名（权重不在此，在 attrs）
    std::vector<std::string> outputs;          // 输出 operand 名（假定单输出为主）
    std::vector<std::vector<int> > output_shapes; // 每个输出的解析后 shape
    std::map<std::string, Pt2Param> params;    // 已命名参数
    std::map<std::string, Pt2Attr> attrs;     // 内联权重
};

struct Pt2Graph
{
    std::vector<Pt2Node> nodes;
    std::vector<Pt2IO> inputs;  // user_input
    std::vector<Pt2IO> outputs; // user_output
};

// 解析中间 zip（规范 pnnx IR JSON + data/N 权重）-> Pt2Graph。返回 0 成功。
int parse_pt2_zip(const std::string& pt2path, Pt2Graph& g);

} // namespace pnnx

#endif // PNNX_LOAD_PT2_PARSE_H
