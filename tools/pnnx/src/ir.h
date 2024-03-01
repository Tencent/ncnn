// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef PNNX_IR_H
#define PNNX_IR_H

#include <limits.h>
#include <complex>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#if BUILD_TORCH2PNNX
namespace torch {
namespace jit {
struct Value;
struct Node;
} // namespace jit
} // namespace torch
namespace at {
class Tensor;
}
#endif // BUILD_TORCH2PNNX

#if BUILD_ONNX2PNNX
namespace onnx {
class TensorProto;
class ValueInfoProto;
} // namespace onnx
#endif // BUILD_ONNX2PNNX

namespace pnnx {

class Parameter
{
public:
    Parameter()
        : type(0)
    {
    }
    Parameter(bool _b)
        : type(1), b(_b)
    {
    }
    Parameter(int _i)
        : type(2), i(_i)
    {
    }
    Parameter(long _l)
        : type(2)
    {
        if (_l == std::numeric_limits<long>::max()) _l = INT_MAX;
        if (_l == std::numeric_limits<long>::min()) _l = INT_MIN;
        i = (int)_l;
    }
    Parameter(long long _l)
        : type(2)
    {
        if (_l == std::numeric_limits<long long>::max()) _l = INT_MAX;
        if (_l == std::numeric_limits<long long>::min()) _l = INT_MIN;
        i = (int)_l;
    }
    Parameter(float _f)
        : type(3), f(_f)
    {
    }
    Parameter(double _d)
        : type(3), f((float)_d)
    {
    }
    Parameter(const char* _s)
        : type(4), s(_s)
    {
    }
    Parameter(const std::string& _s)
        : type(4), s(_s)
    {
    }
    Parameter(const std::initializer_list<int>& _ai)
        : type(5), ai(_ai)
    {
    }
    Parameter(const std::initializer_list<int64_t>& _ai)
        : type(5)
    {
        for (const auto& x : _ai)
        {
            int64_t _l = x;
            if (_l == std::numeric_limits<int64_t>::max()) _l = INT_MAX;
            if (_l == std::numeric_limits<int64_t>::min()) _l = INT_MIN;
            ai.push_back((int)_l);
        }
    }
    Parameter(const std::vector<int>& _ai)
        : type(5), ai(_ai)
    {
    }
    Parameter(const std::vector<int64_t>& _ai)
        : type(5)
    {
        for (const auto& x : _ai)
        {
            int64_t _l = x;
            if (_l == std::numeric_limits<int64_t>::max()) _l = INT_MAX;
            if (_l == std::numeric_limits<int64_t>::min()) _l = INT_MIN;
            ai.push_back((int)_l);
        }
    }
    Parameter(const std::initializer_list<float>& _af)
        : type(6), af(_af)
    {
    }
    Parameter(const std::initializer_list<double>& _af)
        : type(6)
    {
        for (const auto& x : _af)
            af.push_back((float)x);
    }
    Parameter(const std::vector<float>& _af)
        : type(6), af(_af)
    {
    }
    Parameter(const std::vector<double>& _af)
        : type(6)
    {
        for (const auto& x : _af)
            af.push_back((float)x);
    }
    Parameter(const std::initializer_list<const char*>& _as)
        : type(7)
    {
        for (const auto& x : _as)
            as.push_back(std::string(x));
    }
    Parameter(const std::initializer_list<std::string>& _as)
        : type(7), as(_as)
    {
    }
    Parameter(const std::vector<std::string>& _as)
        : type(7), as(_as)
    {
    }
    Parameter(const std::complex<float>& _c)
        : type(10), c(_c)
    {
    }
    Parameter(const std::complex<double>& _c)
        : type(10), c(_c)
    {
    }
    Parameter(const std::initializer_list<std::complex<float> >& _ac)
        : type(11), ac(_ac)
    {
    }
    Parameter(const std::initializer_list<std::complex<double> >& _ac)
        : type(11)
    {
        for (const auto& x : _ac)
            ac.push_back(std::complex<float>(x));
    }
    Parameter(const std::vector<std::complex<float> >& _ac)
        : type(11), ac(_ac)
    {
    }
    Parameter(const std::vector<std::complex<double> >& _ac)
        : type(11)
    {
        for (const auto& x : _ac)
            ac.push_back(std::complex<float>(x));
    }

#if BUILD_TORCH2PNNX
    Parameter(const torch::jit::Node* value_node);
    Parameter(const torch::jit::Value* value);
#endif // BUILD_TORCH2PNNX

    static Parameter parse_from_string(const std::string& value);
    static std::string encode_to_string(const Parameter& param);

    // 0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others 10=c 11=ac
    int type;

    // value
    bool b;
    int i;
    float f;
    std::complex<float> c;
    std::vector<int> ai;
    std::vector<float> af;
    std::vector<std::complex<float> > ac;

    // keep std::string typed member the last for cross cxxabi compatibility
    std::string s;
    std::vector<std::string> as;
};

bool operator==(const Parameter& lhs, const Parameter& rhs);

class Attribute
{
public:
    Attribute()
        : type(0)
    {
    }

#if BUILD_TORCH2PNNX
    Attribute(const at::Tensor& t);
#endif
#if BUILD_ONNX2PNNX
    Attribute(const onnx::TensorProto& t);
#endif

    Attribute(const std::initializer_list<int>& shape, const std::vector<float>& t);

    size_t elemsize() const;
    int elemcount() const;

    // convenient routines for manipulate fp32/fp16 weight
    std::vector<float> get_float32_data() const;
    void set_float32_data(const std::vector<float>& data);

    // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool 10=c64 11=c128 12=c32
    int type;
    std::vector<int> shape;

    std::vector<char> data;

    std::map<std::string, Parameter> params;
};

bool operator==(const Attribute& lhs, const Attribute& rhs);

// concat two attributes along the first axis
Attribute operator+(const Attribute& a, const Attribute& b);

class Operator;
class Operand
{
public:
    void remove_consumer(const Operator* c);

    Operator* producer;
    std::vector<Operator*> consumers;

    // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool 10=c64 11=c128 12=c32
    int type;
    std::vector<int> shape;

    // keep std::string typed member the last for cross cxxabi compatibility
    std::string name;

    std::map<std::string, Parameter> params;

private:
    friend class Graph;
    Operand()
    {
        type = 0;
    }
};

class Operator
{
public:
    bool has_param(const std::string& key) const;
    bool has_attr(const std::string& key) const;
    bool has_input(const std::string& key) const;
    Operand* named_input(const std::string& key);
    const Operand* named_input(const std::string& key) const;

    std::vector<Operand*> inputs;
    std::vector<Operand*> outputs;

    // keep std::string typed member the last for cross cxxabi compatibility
    std::string type;
    std::string name;

    std::vector<std::string> inputnames;
    std::map<std::string, Parameter> params;
    std::map<std::string, Attribute> attrs;

private:
    friend class Graph;
    Operator()
    {
    }
};

class Graph
{
public:
    Graph();
    ~Graph();

    int load(const std::string& parampath, const std::string& binpath);
    int save(const std::string& parampath, const std::string& binpath);

    int python(const std::string& pypath, const std::string& binpath);

    int parse(const std::string& param);

    Operator* new_operator(const std::string& type, const std::string& name);

    Operator* new_operator_before(const std::string& type, const std::string& name, const Operator* cur);

    Operator* new_operator_after(const std::string& type, const std::string& name, const Operator* cur);

#if BUILD_TORCH2PNNX
    Operand* new_operand(const torch::jit::Value* v);
#endif
#if BUILD_ONNX2PNNX
    Operand* new_operand(const onnx::ValueInfoProto& value);
#endif

    Operand* new_operand(const std::string& name);

    Operand* get_operand(const std::string& name);
    const Operand* get_operand(const std::string& name) const;

    std::vector<Operator*> ops;
    std::vector<Operand*> operands;

private:
    Graph(const Graph& rhs);
    Graph& operator=(const Graph& rhs);
};

} // namespace pnnx

#endif // PNNX_IR_H
