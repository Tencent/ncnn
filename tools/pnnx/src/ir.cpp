// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ir.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <stack>

#include "storezip.h"
#include "utils.h"

namespace pnnx {

static bool type_is_integer(int type)
{
    if (type == 1) return false;
    if (type == 2) return false;
    if (type == 3) return false;
    if (type == 4) return true;
    if (type == 5) return true;
    if (type == 6) return true;
    if (type == 7) return true;
    if (type == 8) return true;
    if (type == 9) return true;
    if (type == 10) return false;
    if (type == 11) return false;
    if (type == 12) return false;
    if (type == 13) return false;
    return false;
}

static const char* type_to_string(int type)
{
    if (type == 1) return "f32";
    if (type == 2) return "f64";
    if (type == 3) return "f16";
    if (type == 4) return "i32";
    if (type == 5) return "i64";
    if (type == 6) return "i16";
    if (type == 7) return "i8";
    if (type == 8) return "u8";
    if (type == 9) return "bool";
    if (type == 10) return "c64";
    if (type == 11) return "c128";
    if (type == 12) return "c32";
    if (type == 13) return "bf16";
    return "null";
}

static const char* type_to_numpy_string(int type)
{
    if (type == 1) return "float32";
    if (type == 2) return "float64";
    if (type == 3) return "float16";
    if (type == 4) return "int32";
    if (type == 5) return "int64";
    if (type == 6) return "int16";
    if (type == 7) return "int8";
    if (type == 8) return "uint8";
    if (type == 9) return "bool";
    if (type == 10) return "csingle";
    if (type == 11) return "cdouble";
    if (type == 12) return "chalf";
    if (type == 13) return "bfloat16";
    return "null";
}

static const char* type_to_dtype_string(int type)
{
    if (type == 1) return "torch.float";
    if (type == 2) return "torch.double";
    if (type == 3) return "torch.half";
    if (type == 4) return "torch.int";
    if (type == 5) return "torch.long";
    if (type == 6) return "torch.short";
    if (type == 7) return "torch.int8";
    if (type == 8) return "torch.uint8";
    if (type == 9) return "torch.bool";
    if (type == 10) return "torch.complex64";
    if (type == 11) return "torch.complex128";
    if (type == 12) return "torch.complex32";
    if (type == 13) return "torch.bfloat16";
    return "null";
}

static size_t type_to_elemsize(int type)
{
    if (type == 1) return 4;
    if (type == 2) return 8;
    if (type == 3) return 2;
    if (type == 4) return 4;
    if (type == 5) return 8;
    if (type == 6) return 2;
    if (type == 7) return 1;
    if (type == 8) return 1;
    if (type == 9) return 1;
    if (type == 10) return 8;
    if (type == 11) return 16;
    if (type == 12) return 4;
    if (type == 13) return 2;
    return 0; // null
}

static int string_to_type(const char* s)
{
    if (strcmp(s, "f32") == 0) return 1;
    if (strcmp(s, "f64") == 0) return 2;
    if (strcmp(s, "f16") == 0) return 3;
    if (strcmp(s, "i32") == 0) return 4;
    if (strcmp(s, "i64") == 0) return 5;
    if (strcmp(s, "i16") == 0) return 6;
    if (strcmp(s, "i8") == 0) return 7;
    if (strcmp(s, "u8") == 0) return 8;
    if (strcmp(s, "bool") == 0) return 9;
    if (strcmp(s, "c64") == 0) return 10;
    if (strcmp(s, "c128") == 0) return 11;
    if (strcmp(s, "c32") == 0) return 12;
    if (strcmp(s, "bf16") == 0) return 13;
    return 0; // null
}

bool operator==(const Parameter& lhs, const Parameter& rhs)
{
    if (lhs.type != rhs.type)
        return false;

    if (lhs.type == 0)
        return true;

    if (lhs.type == 1 && lhs.b == rhs.b)
        return true;

    if (lhs.type == 2 && lhs.i == rhs.i)
        return true;

    if (lhs.type == 3 && lhs.f == rhs.f)
        return true;

    if (lhs.type == 4 && lhs.s == rhs.s)
        return true;

    if (lhs.type == 5 && lhs.ai == rhs.ai)
        return true;

    if (lhs.type == 6 && lhs.af == rhs.af)
        return true;

    if (lhs.type == 7 && lhs.as == rhs.as)
        return true;

    if (lhs.type == 10 && lhs.c == rhs.c)
        return true;

    if (lhs.type == 11 && lhs.ac == rhs.ac)
        return true;

    return false;
}

Attribute::Attribute(const std::initializer_list<int>& _shape, const std::vector<float>& t)
{
    type = 1;
    shape = _shape;

    if (shape.size() > 0)
    {
        data.resize(elemcount() * type_to_elemsize(type));
        memcpy((void*)data.data(), (const void*)t.data(), data.size());
    }
}

size_t Attribute::elemsize() const
{
    return type_to_elemsize(type);
}

int Attribute::elemcount() const
{
    if (shape.empty())
        return 0;

    int size = shape[0];
    for (size_t i = 1; i < shape.size(); i++)
    {
        size *= shape[i];
    }

    return size;
}

std::vector<float> Attribute::get_float32_data() const
{
    std::vector<float> v(elemcount());

    if (type == 1)
    {
        memcpy((void*)v.data(), (const void*)data.data(), data.size());
    }
    else if (type == 2)
    {
        // f64
        const double* p = (const double*)data.data();
        for (size_t i = 0; i < v.size(); i++)
        {
            v[i] = float(p[i]);
        }
    }
    else if (type == 3)
    {
        // f16
        const unsigned short* p = (const unsigned short*)data.data();
        for (size_t i = 0; i < v.size(); i++)
        {
            v[i] = float16_to_float32(p[i]);
        }
    }
    else
    {
        fprintf(stderr, "cannot convert type %d to float32 data\n", type);
    }

    return v;
}

void Attribute::set_float32_data(const std::vector<float>& newdata)
{
    data.resize(newdata.size() * elemsize());

    if (type == 1)
    {
        memcpy((void*)data.data(), (const void*)newdata.data(), data.size());
    }
    else if (type == 2)
    {
        // f64
        double* p = (double*)data.data();
        for (size_t i = 0; i < newdata.size(); i++)
        {
            p[i] = newdata[i];
        }
    }
    else if (type == 3)
    {
        // f16
        unsigned short* p = (unsigned short*)data.data();
        for (size_t i = 0; i < newdata.size(); i++)
        {
            p[i] = float32_to_float16(newdata[i]);
        }
    }
    else
    {
        fprintf(stderr, "cannot convert float32 data to type %d\n", type);
    }
}

bool operator==(const Attribute& lhs, const Attribute& rhs)
{
    if (lhs.type != rhs.type)
        return false;

    if (lhs.type == 0)
        return true;

    if (lhs.shape != rhs.shape)
        return false;

    if (lhs.data != rhs.data)
        return false;

    return true;
}

Attribute operator+(const Attribute& a, const Attribute& b)
{
    Attribute c;

    if (a.type != b.type)
    {
        fprintf(stderr, "concat attribute type mismatch\n");
        return c;
    }

    if (a.shape.size() != b.shape.size())
    {
        fprintf(stderr, "concat attribute shape rank mismatch\n");
        return c;
    }

    for (int i = 1; i < (int)a.shape.size(); i++)
    {
        if (a.shape[i] != b.shape[i])
        {
            fprintf(stderr, "concat attribute shape mismatch\n");
            return c;
        }
    }

    c.type = a.type;
    c.shape = a.shape;
    c.shape[0] += b.shape[0]; // concat the first dim

    c.data.resize(a.data.size() + b.data.size());
    memcpy(c.data.data(), a.data.data(), a.data.size());
    memcpy(c.data.data() + a.data.size(), b.data.data(), b.data.size());

    return c;
}

Parameter Parameter::parse_from_string(const std::string& value)
{
    if (value.find('%') != std::string::npos)
    {
        Parameter p;
        p.type = 4;
        p.s = value;
        return p;
    }

    Parameter p;
    p.type = 0;

    if (value == "None" || value == "()" || value == "[]")
    {
        return p;
    }

    if (value == "True" || value == "False")
    {
        // bool
        p.type = 1;
        p.b = value == "True";
        return p;
    }

    if (value[0] == '(' || value[0] == '[')
    {
        // list
        std::string lc = value.substr(1, value.size() - 2);
        std::istringstream lcss(lc);

        while (!lcss.eof())
        {
            std::string elem;
            std::getline(lcss, elem, ',');

            if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) || (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9')))
            {
                // string
                p.type = 7;
                p.as.push_back(elem);
            }
            else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos)
            {
                // float
                p.type = 6;
                p.af.push_back(std::stof(elem));
            }
            else
            {
                // integer
                p.type = 5;
                p.ai.push_back(std::stoi(elem));
            }
        }
        return p;
    }

    if ((value[0] != '-' && (value[0] < '0' || value[0] > '9')) || (value[0] == '-' && (value[1] < '0' || value[1] > '9')))
    {
        // string
        p.type = 4;
        p.s = value;
        return p;
    }

    if (value.find('.') != std::string::npos || value.find('e') != std::string::npos)
    {
        // float
        p.type = 3;
        p.f = std::stof(value);
        return p;
    }

    // integer
    p.type = 2;
    p.i = std::stoi(value);
    return p;
}

std::string Parameter::encode_to_string(const Parameter& param)
{
    if (param.type == 0)
    {
        return std::string("None");
    }
    if (param.type == 1)
    {
        if (param.b)
            return std::string("True");
        else
            return std::string("False");
    }
    if (param.type == 2)
    {
        return std::to_string(param.i);
    }
    if (param.type == 3)
    {
        return float_to_string(param.f);
    }
    if (param.type == 4)
    {
        return param.s;
    }
    if (param.type == 5)
    {
        std::string s("(");
        for (size_t i = 0; i < param.ai.size(); i++)
        {
            s += std::to_string(param.ai[i]);
            if (i + 1 != param.ai.size())
                s += std::string(",");
        }
        s += std::string(")");
        return s;
    }
    if (param.type == 6)
    {
        std::string s("(");
        for (size_t i = 0; i < param.af.size(); i++)
        {
            s += float_to_string(param.af[i]);
            if (i + 1 != param.af.size())
                s += std::string(",");
        }
        s += std::string(")");
        return s;
    }
    if (param.type == 7)
    {
        std::string s("(");
        for (size_t i = 0; i < param.as.size(); i++)
        {
            s += param.as[i];
            if (i + 1 != param.as.size())
                s += std::string(",");
        }
        s += std::string(")");
        return s;
    }
    if (param.type == 10)
    {
        char buf[128];
        sprintf(buf, "%e+%ej", param.c.real(), param.c.imag());
        return std::string(buf);
    }
    if (param.type == 11)
    {
        std::string s("(");
        for (size_t i = 0; i < param.ac.size(); i++)
        {
            char buf[128];
            sprintf(buf, "%e+%ej", param.ac[i].real(), param.ac[i].imag());
            s += std::string(buf);
            if (i + 1 != param.ac.size())
                s += std::string(",");
        }
        s += std::string(")");
        return s;
    }

    fprintf(stderr, "unknown parameter type %d\n", param.type);
    return std::string();
}

bool Operator::has_param(const std::string& key) const
{
    return params.find(key) != params.end();
}

bool Operator::has_attr(const std::string& key) const
{
    return attrs.find(key) != attrs.end();
}

bool Operator::has_input(const std::string& key) const
{
    return std::find(inputnames.begin(), inputnames.end(), key) != inputnames.end();
}

Operand* Operator::named_input(const std::string& key)
{
    for (size_t i = 0; i < inputnames.size(); i++)
    {
        if (inputnames[i] == key)
            return inputs[i];
    }

    return 0;
}

const Operand* Operator::named_input(const std::string& key) const
{
    for (size_t i = 0; i < inputnames.size(); i++)
    {
        if (inputnames[i] == key)
            return inputs[i];
    }

    return 0;
}

Graph::Graph()
{
}

Graph::~Graph()
{
    for (auto x : ops)
        delete x;

    for (auto x : operands)
        delete x;

    ops.clear();
    operands.clear();
}

Graph::Graph(const Graph& /*rhs*/)
{
}

Graph& Graph::operator=(const Graph& /*rhs*/)
{
    return *this;
}

static void load_parameter(Operator* op, const std::string& key, const std::string& value)
{
    op->params[key] = Parameter::parse_from_string(value);
}

static void load_input_key(Operator* op, const std::string& key, const std::string& value)
{
    op->inputnames.resize(op->inputs.size());

    for (size_t i = 0; i < op->inputs.size(); i++)
    {
        const Operand* oprand = op->inputs[i];
        if (oprand->name == value)
        {
            op->inputnames[i] = key;
            break;
        }
    }
}

static void load_shape(Operator* op, const std::string& key, const std::string& value)
{
    Operand* operand = 0;
    for (auto r : op->inputs)
    {
        if (r->name == key)
        {
            operand = r;
            break;
        }
    }

    if (!operand)
    {
        for (auto r : op->outputs)
        {
            if (r->name == key)
            {
                operand = r;
                break;
            }
        }
    }

    if (!operand)
    {
        fprintf(stderr, "no such operand %s for operator %s\n", key.c_str(), op->name.c_str());
        return;
    }

    // type
    std::string typestr = value.substr(value.find_last_of(')') + 1);
    operand->type = string_to_type(typestr.c_str());

    // shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);

    operand->shape.clear();
    while (!lcss.eof())
    {
        std::string elem;
        std::getline(lcss, elem, ',');

        if (elem == "?")
        {
            operand->shape.push_back(-1);
        }
        else if (elem[0] == '%')
        {
            // encode %abc as symbolic tag
            operand->shape.push_back(-233);
            int index = operand->shape.size() - 1;
            std::string key = elem.substr(1);
            operand->params[std::string("__shape__") + std::to_string(index)] = key;
        }
        else
        {
            int i = std::stoi(elem);
            operand->shape.push_back(i);
        }
    }
}

static void load_attribute(Operator* op, const std::string& key, const std::string& value, StoreZipReader& szr)
{
    Attribute& a = op->attrs[key];

    // type
    std::string typestr = value.substr(value.find_last_of(')') + 1);
    a.type = string_to_type(typestr.c_str());

    if (a.type == 0)
        return;

    // shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);

    a.shape.clear();
    while (!lcss.eof())
    {
        std::string elem;
        std::getline(lcss, elem, ',');

        int i = std::stoi(elem);
        a.shape.push_back(i);
    }

    if (a.shape.empty())
        return;

    // data
    size_t size = 1;
    for (int i : a.shape)
    {
        size *= i;
    }

    size_t bytesize = size * type_to_elemsize(a.type);

    std::string filename = op->name + "." + key;

    size_t filesize = szr.get_file_size(filename);

    if (filesize == 0)
    {
        // no such file
        return;
    }

    if (filesize != bytesize)
    {
        fprintf(stderr, "file size not match expect %lu but got %lu\n", bytesize, filesize);
    }

    a.data.resize(bytesize);
    szr.read_file(filename, (char*)a.data.data());
}

int Graph::load(const std::string& parampath, const std::string& binpath)
{
    std::ifstream is(parampath, std::ios::in | std::ios::binary);
    if (!is.good())
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    StoreZipReader szr;
    if (szr.open(binpath) != 0)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    int magic = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> magic;
    }

    int operator_count = 0;
    int operand_count = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> operator_count >> operand_count;
    }

    for (int i = 0; i < operator_count; i++)
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        std::string type;
        std::string name;
        int input_count = 0;
        int output_count = 0;

        iss >> type >> name >> input_count >> output_count;

        Operator* op = new_operator(type, name);

        for (int j = 0; j < input_count; j++)
        {
            std::string operand_name;
            iss >> operand_name;

            Operand* r = get_operand(operand_name);
            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }

        for (int j = 0; j < output_count; j++)
        {
            std::string operand_name;
            iss >> operand_name;

            Operand* r = new_operand(operand_name);
            r->producer = op;
            op->outputs.push_back(r);
        }

        // key=value
        while (!iss.eof())
        {
            std::string param;
            iss >> param;

            std::string key;
            std::string value;
            std::istringstream pss(param);
            std::getline(pss, key, '=');
            std::getline(pss, value);

            if (key[0] == '@')
            {
                // attribute
                load_attribute(op, key.substr(1), value, szr);
            }
            else if (key[0] == '$')
            {
                // operand input key
                load_input_key(op, key.substr(1), value);
            }
            else if (key[0] == '#')
            {
                // operand shape
                load_shape(op, key.substr(1), value);
            }
            else
            {
                // parameter
                load_parameter(op, key, value);
            }
        }
    }

    return 0;
}

int Graph::save(const std::string& parampath, const std::string& binpath)
{
    FILE* paramfp = fopen(parampath.c_str(), "wb");
    if (!paramfp)
    {
        fprintf(stderr, "fopen %s failed\n", parampath.c_str());
        return -1;
    }

    StoreZipWriter szw;
    if (szw.open(binpath) != 0)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    // magic
    fprintf(paramfp, "7767517\n");

    // op count and oprand count
    fprintf(paramfp, "%d %d\n", (int)ops.size(), (int)operands.size());

    for (const Operator* op : ops)
    {
        fprintf(paramfp, "%-24s %-24s %d %d", op->type.c_str(), op->name.c_str(), (int)op->inputs.size(), (int)op->outputs.size());

        for (const Operand* oprand : op->inputs)
        {
            fprintf(paramfp, " %s", oprand->name.c_str());
        }

        for (const Operand* oprand : op->outputs)
        {
            fprintf(paramfp, " %s", oprand->name.c_str());
        }

        for (const auto& it : op->params)
        {
            fprintf(paramfp, " %s=", it.first.c_str());

            const Parameter& param = it.second;
            std::string s = Parameter::encode_to_string(param);
            fprintf(paramfp, "%s", s.c_str());
        }

        for (const auto& it : op->attrs)
        {
            fprintf(paramfp, " @%s=", it.first.c_str());

            const Attribute& attr = it.second;
            fprintf(paramfp, "(");
            for (int i = 0; i < (int)attr.shape.size() - 1; i++)
            {
                fprintf(paramfp, "%d,", attr.shape[i]);
            }
            if (attr.shape.size() > 0)
                fprintf(paramfp, "%d", attr.shape[attr.shape.size() - 1]);
            fprintf(paramfp, ")");

            fprintf(paramfp, type_to_string(attr.type));

            std::string filename = op->name + "." + it.first;
            szw.write_file(filename, attr.data.data(), attr.data.size());
        }

        if (op->inputnames.size() == op->inputs.size())
        {
            for (size_t i = 0; i < op->inputs.size(); i++)
            {
                if (op->inputnames[i].empty())
                    continue;

                const Operand* oprand = op->inputs[i];
                fprintf(paramfp, " $%s=%s", op->inputnames[i].c_str(), oprand->name.c_str());
            }
        }

        for (const Operand* oprand : op->inputs)
        {
            if (oprand->shape.empty())
                continue;

            fprintf(paramfp, " #%s=", oprand->name.c_str());

            fprintf(paramfp, "(");
            for (int i = 0; i < (int)oprand->shape.size() - 1; i++)
            {
                if (oprand->shape[i] == -1)
                    fprintf(paramfp, "?,");
                else
                    fprintf(paramfp, "%d,", oprand->shape[i]);
            }
            if (oprand->shape.size() > 0)
            {
                if (oprand->shape[oprand->shape.size() - 1] == -1)
                    fprintf(paramfp, "?");
                else
                    fprintf(paramfp, "%d", oprand->shape[oprand->shape.size() - 1]);
            }
            fprintf(paramfp, ")");

            fprintf(paramfp, type_to_string(oprand->type));
        }

        for (const Operand* oprand : op->outputs)
        {
            if (oprand->shape.empty())
                continue;

            fprintf(paramfp, " #%s=", oprand->name.c_str());

            fprintf(paramfp, "(");
            for (int i = 0; i < (int)oprand->shape.size() - 1; i++)
            {
                if (oprand->shape[i] == -1)
                    fprintf(paramfp, "?,");
                else
                    fprintf(paramfp, "%d,", oprand->shape[i]);
            }
            if (oprand->shape.size() > 0)
            {
                if (oprand->shape[oprand->shape.size() - 1] == -1)
                    fprintf(paramfp, "?");
                else
                    fprintf(paramfp, "%d", oprand->shape[oprand->shape.size() - 1]);
            }
            fprintf(paramfp, ")");

            fprintf(paramfp, type_to_string(oprand->type));
        }

        fprintf(paramfp, "\n");
    }

    fclose(paramfp);

    return 0;
}

static std::string sanitize_identifier(const std::string& s)
{
    std::string ss = s;
    for (size_t i = 0; i < ss.size(); i++)
    {
        if (ss[i] == '.' || ss[i] == ':' || ss[i] == '/')
            ss[i] = '_';
    }

    return ss;
}

static bool token_is_complex(const std::string& t)
{
    // 2.000000e+00+3.000000e+00j
    if (t[t.size() - 1] != 'j')
        return false;

    return true;
}

static bool token_is_literal(const std::string& t)
{
    if (token_is_complex(t))
        return true;

    std::istringstream iss(t);
    float f;
    iss >> std::noskipws >> f;
    return iss.eof() && !iss.fail();
}

static std::string expand_expression(const Operator* op)
{
    std::string expr = op->params.at("expr").s;

    // split into tokens
    std::vector<std::string> tokens;
    {
        std::string t;
        for (size_t i = 0; i < expr.size(); i++)
        {
            char ch = expr[i];

            if (ch == '[') // list
            {
                t += ch;
                tokens.push_back(t);
                t.clear();
            }
            else if (ch == '(' || ch == ')' || ch == ',' || ch == ']')
            {
                if (!t.empty())
                {
                    tokens.push_back(t);
                    t.clear();
                }
            }
            else
            {
                t += ch;
            }
        }

        if (!t.empty())
        {
            tokens.push_back(t);
        }
    }

    // scan and stack
    std::stack<std::string> exprstack;
    for (int i = (int)tokens.size() - 1; i >= 0; i--)
    {
        const std::string& t = tokens[i];

        if (t == "size")
        {
            std::string a = exprstack.top();
            exprstack.pop();

            if (exprstack.empty())
            {
                std::string r = a + ".shape";
                exprstack.push(r);
            }
            else
            {
                std::string b = exprstack.top();
                exprstack.pop();

                std::string r = a + ".size(" + b + ")";
                exprstack.push(r);
            }
        }
        else if (t == "int"
                 || t == "abs"
                 || t == "acos"
                 || t == "acosh"
                 || t == "asin"
                 || t == "asinh"
                 || t == "atan"
                 || t == "atanh"
                 || t == "ceil"
                 || t == "cos"
                 || t == "cosh"
                 || t == "erf"
                 || t == "exp"
                 || t == "floor"
                 || t == "log"
                 || t == "log10"
                 || t == "neg"
                 || t == "reciprocal"
                 || t == "round"
                 || t == "rsqrt"
                 || t == "sign"
                 || t == "sin"
                 || t == "sinh"
                 || t == "sqrt"
                 || t == "square"
                 || t == "tan"
                 || t == "tanh"
                 || t == "trunc"
                 || t == "torch.bool"
                 || t == "torch.float"
                 || t == "torch.long")
        {
            std::string unaryop = t;
            if (t == "int") unaryop = ""; // but the explicit int() causes troubles in tracing
            if (t == "abs") unaryop = "torch.abs";
            if (t == "acos") unaryop = "torch.acos";
            if (t == "acosh") unaryop = "torch.acosh";
            if (t == "asin") unaryop = "torch.asin";
            if (t == "asinh") unaryop = "torch.asinh";
            if (t == "atan") unaryop = "torch.atan";
            if (t == "atanh") unaryop = "torch.atanh";
            if (t == "ceil") unaryop = "torch.ceil";
            if (t == "cos") unaryop = "torch.cos";
            if (t == "cosh") unaryop = "torch.cosh";
            if (t == "erf") unaryop = "torch.erf";
            if (t == "exp") unaryop = "torch.exp";
            if (t == "floor") unaryop = "torch.floor";
            if (t == "log") unaryop = "torch.log";
            if (t == "log10") unaryop = "torch.log10";
            if (t == "neg") unaryop = "-";
            if (t == "reciprocal") unaryop = "torch.reciprocal";
            if (t == "round") unaryop = "torch.round";
            if (t == "rsqrt") unaryop = "torch.rsqrt";
            if (t == "sign") unaryop = "torch.sign";
            if (t == "sin") unaryop = "torch.sin";
            if (t == "sinh") unaryop = "torch.sinh";
            if (t == "sqrt") unaryop = "torch.sqrt";
            if (t == "square") unaryop = "torch.square";
            if (t == "tan") unaryop = "torch.tan";
            if (t == "tanh") unaryop = "torch.tanh";
            if (t == "trunc") unaryop = "torch.trunc";

            std::string a = exprstack.top();
            exprstack.pop();

            std::string r = unaryop + "(" + a + ")";
            exprstack.push(r);
        }
        else if (t == "atan2"
                 || t == "fmod"
                 || t == "max"
                 || t == "maximum"
                 || t == "min"
                 || t == "minimum"
                 || t == "pow"
                 || t == "logaddexp")
        {
            std::string binaryop;
            if (t == "atan2") binaryop = "torch.atan2";
            if (t == "fmod") binaryop = "torch.fmod";
            if (t == "max") binaryop = "torch.max";
            if (t == "maximum") binaryop = "torch.maximum";
            if (t == "min") binaryop = "torch.min";
            if (t == "minimum") binaryop = "torch.minimum";
            if (t == "pow") binaryop = "torch.pow";
            if (t == "logaddexp") binaryop = "torch.logaddexp";

            std::string a = exprstack.top();
            exprstack.pop();
            std::string b = exprstack.top();
            exprstack.pop();

            if (t == "max" || t == "min")
            {
                if (token_is_literal(a))
                    a = std::string("torch.tensor(") + a + ")";
                if (token_is_literal(b))
                    b = std::string("torch.tensor(") + b + ")";
            }

            std::string r = binaryop + "(" + a + ", " + b + ")";
            exprstack.push(r);
        }
        else if (t == "add"
                 || t == "sub"
                 || t == "mul"
                 || t == "div"
                 || t == "floor_divide"
                 || t == "remainder"
                 || t == "and"
                 || t == "or"
                 || t == "xor"
                 || t == "lshift"
                 || t == "rshift")
        {
            std::string binaryop;
            if (t == "add") binaryop = "+";
            if (t == "sub") binaryop = "-";
            if (t == "mul") binaryop = "*";
            if (t == "div") binaryop = "/";
            if (t == "floor_divide") binaryop = "//";
            if (t == "remainder") binaryop = "%";
            if (t == "and") binaryop = "&";
            if (t == "or") binaryop = "|";
            if (t == "xor") binaryop = "^";
            if (t == "lshift") binaryop = "<<";
            if (t == "rshift") binaryop = ">>";

            std::string a = exprstack.top();
            exprstack.pop();
            std::string b = exprstack.top();
            exprstack.pop();

            std::string r = std::string("(") + a + " " + binaryop + " " + b + ")";
            exprstack.push(r);
        }
        else if (t == "[") // list
        {
            std::vector<std::string> elements;
            while (!exprstack.empty())
            {
                std::string a = exprstack.top();
                exprstack.pop();

                elements.push_back(a);
            }

            std::string r = "[";
            for (int j = 0; j < (int)elements.size() - 1; j++)
            {
                r += elements[j];
                if (j + 1 != (int)elements.size())
                    r += ", ";
            }
            if (!elements.empty())
            {
                r += elements[elements.size() - 1];
            }
            r += "]";

            exprstack.push(r);
        }
        else if (t[0] == '@')
        {
            int input_index = std::stoi(t.substr(1));
            std::string varid = std::string("v_") + sanitize_identifier(op->inputs[input_index]->name);
            exprstack.push(varid);
        }
        else
        {
            // literal
            if (t[t.size() - 1] == 'j')
            {
                // complex
                std::string r = std::string("(") + t + ")";
                exprstack.push(r);
            }
            else
            {
                exprstack.push(t);
            }
        }
    }

    std::string r = exprstack.top();
    exprstack.pop();

    return r;
}

static std::string make_slice_expression(const Operator* op)
{
    // for (size_t j = 0; j < op->inputnames.size(); j++)
    // {
    //     fprintf(stderr, "make_slice_expression %s %s\n", op->inputnames[j].c_str(), op->inputs[j]->name.c_str());
    // }

    std::vector<int> dims;
    if (op->has_param("dims"))
    {
        dims = op->params.at("dims").ai;
    }
    else
    {
        dims.push_back(op->params.at("dim").i);
    }

    std::string pr;
    std::string nr;

    int last_dim = -1;
    const int ndim = (int)dims.size();
    for (int i = 0; i < ndim; i++)
    {
        int dim = dims[i];
        std::string& r = dim < 0 ? nr : pr;

        for (int j = last_dim + 1; j < dim; j++)
        {
            r += ":,";
        }
        last_dim = dim;

        bool is_select = false;
        if (op->has_param("select"))
        {
            int select = op->params.at("select").i;
            if (select != INT_MAX)
            {
                r += std::to_string(select);
                is_select = true;
            }
        }
        if (op->has_param("selects"))
        {
            std::vector<int> selects = op->params.at("selects").ai;
            int select = selects[i];
            if (select != INT_MAX)
            {
                r += std::to_string(select);
                is_select = true;
            }
        }
        if (op->has_input("select"))
        {
            r += std::string("v_") + sanitize_identifier(op->named_input("select")->name);
            is_select = true;
        }
        if (op->has_input("selects"))
        {
            // must be pnnx.SliceIndexes
            const Operator* op_sliceindexes = op->named_input("selects")->producer;
            const std::string& index = op_sliceindexes->params.at("indexes").as[i];
            if (index[0] == '@')
            {
                int selecti = std::stoi(index.substr(1));
                r += std::string("v_") + sanitize_identifier(op_sliceindexes->inputs[selecti]->name);
                is_select = true;
            }
            else
            {
                int select = std::stoi(index);
                if (select != INT_MAX)
                {
                    r += std::to_string(select);
                    is_select = true;
                }
            }
        }

        if (is_select)
        {
            if (i + 1 != ndim)
                r += ',';
            continue;
        }

        if (op->has_param("start"))
        {
            int start = op->params.at("start").i;
            if (start != 0)
                r += std::to_string(start);
        }
        else if (op->has_param("starts"))
        {
            std::vector<int> starts = op->params.at("starts").ai;
            int start = starts[i];
            if (start != 0)
                r += std::to_string(start);
        }
        else if (op->has_input("start"))
        {
            r += std::string("v_") + sanitize_identifier(op->named_input("start")->name);
        }
        else // if (op->has_input("starts"))
        {
            // must be pnnx.SliceIndexes
            const Operator* op_sliceindexes = op->named_input("starts")->producer;
            const std::string& index = op_sliceindexes->params.at("indexes").as[i];
            if (index[0] == '@')
            {
                int starti = std::stoi(index.substr(1));
                r += std::string("v_") + sanitize_identifier(op_sliceindexes->inputs[starti]->name);
            }
            else
            {
                int start = std::stoi(index);
                if (start != 0)
                    r += std::to_string(start);
            }
        }

        r += ':';

        if (op->has_param("end"))
        {
            int end = op->params.at("end").i;
            if (end != INT_MAX)
                r += std::to_string(end);
        }
        else if (op->has_param("ends"))
        {
            std::vector<int> ends = op->params.at("ends").ai;
            int end = ends[i];
            if (end != INT_MAX)
                r += std::to_string(end);
        }
        else if (op->has_input("end"))
        {
            r += std::string("v_") + sanitize_identifier(op->named_input("end")->name);
        }
        else // if (op->has_input("ends"))
        {
            // must be pnnx.SliceIndexes
            const Operator* op_sliceindexes = op->named_input("ends")->producer;
            const std::string& index = op_sliceindexes->params.at("indexes").as[i];
            if (index[0] == '@')
            {
                int endi = std::stoi(index.substr(1));
                r += std::string("v_") + sanitize_identifier(op_sliceindexes->inputs[endi]->name);
            }
            else
            {
                int end = std::stoi(index);
                if (end != INT_MAX)
                    r += std::to_string(end);
            }
        }

        if (op->has_param("step"))
        {
            int step = op->params.at("step").i;
            if (step != 1)
            {
                r += ':';
                r += std::to_string(step);
            }
        }
        else if (op->has_param("steps"))
        {
            std::vector<int> steps = op->params.at("steps").ai;
            int step = steps[i];
            if (step != 1)
            {
                r += ':';
                r += std::to_string(step);
            }
        }
        else if (op->has_input("step"))
        {
            r += ':';
            r += std::string("v_") + sanitize_identifier(op->named_input("step")->name);
        }
        else // if (op->has_input("steps"))
        {
            // must be pnnx.SliceIndexes
            const Operator* op_sliceindexes = op->named_input("steps")->producer;
            const std::string& index = op_sliceindexes->params.at("indexes").as[i];
            if (index[0] == '@')
            {
                int stepi = std::stoi(index.substr(1));
                r += ':';
                r += std::string("v_") + sanitize_identifier(op_sliceindexes->inputs[stepi]->name);
            }
            else
            {
                int step = std::stoi(index);
                if (step != 1)
                {
                    r += ':';
                    r += std::to_string(step);
                }
            }
        }

        if (i + 1 != ndim)
            r += ',';
    }

    if (!pr.empty() && !nr.empty())
        return pr + "...," + nr;

    if (pr.empty() && !nr.empty())
        return std::string("...,") + nr;

    return pr + nr;
}

static std::string make_index_expression(const Operator* op)
{
    fprintf(stderr, "make_index_expression %s\n", op->name.c_str());

    std::string index_expr = op->params.at("expr").s;

    // strip out-most [ ] pair
    index_expr = index_expr.substr(1, index_expr.size() - 2);

    // None,None,   ->   ...,
    bool leading_none = false;
    while (index_expr.substr(0, 5) == "None,")
    {
        leading_none = true;
        index_expr = index_expr.substr(5);
    }
    if (leading_none)
    {
        index_expr = "...," + index_expr;
    }

    return index_expr;
}

int Graph::python(const std::string& pypath, const std::string& pnnxbinpath, const std::vector<std::vector<int64_t> >& input_shapes)
{
    FILE* pyfp = fopen(pypath.c_str(), "wb");
    if (!pyfp)
    {
        fprintf(stderr, "fopen %s failed\n", pypath.c_str());
        return -1;
    }

    fprintf(pyfp, "import os\n");
    fprintf(pyfp, "import numpy as np\n");
    fprintf(pyfp, "import tempfile, zipfile\n");
    fprintf(pyfp, "import torch\n");
    fprintf(pyfp, "import torch.nn as nn\n");
    fprintf(pyfp, "import torch.nn.functional as F\n");
    fprintf(pyfp, "try:\n");
    fprintf(pyfp, "    import torchvision\n");
    fprintf(pyfp, "    import torchaudio\n");
    fprintf(pyfp, "except:\n");
    fprintf(pyfp, "    pass\n");

    fprintf(pyfp, "\n");

    fprintf(pyfp, "class Model(nn.Module):\n");
    fprintf(pyfp, "    def __init__(self):\n");
    fprintf(pyfp, "        super(Model, self).__init__()\n");

    fprintf(pyfp, "\n");

    // module
    {
        for (const Operator* op : ops)
        {
            if (op->type.substr(0, 3) != "nn." && op->type.substr(0, 16) != "torchvision.ops.")
                continue;

            fprintf(pyfp, "        self.%s = %s(", sanitize_identifier(op->name).c_str(), op->type.c_str());

            int param_count = op->params.size();
            if (op->type == "nn.quantized.Conv2d" || op->type == "nn.quantized.Linear")
            {
                param_count -= 2; // ignore scale and zero_point
            }

            int param_index = 0;
            for (const auto& it : op->params)
            {
                if (op->type == "nn.quantized.Conv2d" || op->type == "nn.quantized.Linear")
                {
                    if (it.first == "scale" || it.first == "zero_point")
                        continue;
                }

                fprintf(pyfp, "%s=", it.first.c_str());

                const Parameter& param = it.second;
                if (param.type == 0)
                {
                    fprintf(pyfp, "None");
                }
                if (param.type == 1)
                {
                    if (param.b)
                        fprintf(pyfp, "True");
                    else
                        fprintf(pyfp, "False");
                }
                if (param.type == 2)
                {
                    fprintf(pyfp, "%d", param.i);
                }
                if (param.type == 3)
                {
                    std::string fs = float_to_string(param.f);
                    fprintf(pyfp, "%s", fs.c_str());
                }
                if (param.type == 4)
                {
                    if (param.s.substr(0, 6) == "torch.")
                    {
                        fprintf(pyfp, "%s", param.s.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "\'%s\'", param.s.c_str());
                    }
                }
                if (param.type == 5)
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < param.ai.size(); i++)
                    {
                        if ((op->type == "nn.AdaptiveAvgPool2d"
                                || op->type == "nn.AdaptiveAvgPool3d"
                                || op->type == "nn.AdaptiveMaxPool2d"
                                || op->type == "nn.AdaptiveMaxPool3d")
                                && it.first == "output_size" && param.ai[i] == 0)
                        {
                            fprintf(pyfp, "None");
                        }
                        else
                        {
                            fprintf(pyfp, "%d", param.ai[i]);
                        }
                        if (i + 1 != param.ai.size() || param.ai.size() == 1)
                            fprintf(pyfp, ",");
                    }
                    fprintf(pyfp, ")");
                }
                if (param.type == 6)
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < param.af.size(); i++)
                    {
                        std::string afs = float_to_string(param.af[i]);
                        fprintf(pyfp, "%s", afs.c_str());
                        if (i + 1 != param.af.size() || param.af.size() == 1)
                            fprintf(pyfp, ",");
                    }
                    fprintf(pyfp, ")");
                }
                if (param.type == 7)
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < param.as.size(); i++)
                    {
                        if (param.as[i].substr(0, 6) == "torch.")
                        {
                            fprintf(pyfp, "%s", param.as[i].c_str());
                        }
                        else
                        {
                            fprintf(pyfp, "\'%s\'", param.as[i].c_str());
                        }
                        if (i + 1 != param.as.size() || param.as.size() == 1)
                            fprintf(pyfp, ",");
                    }
                    fprintf(pyfp, ")");
                }

                param_index++;
                if (param_index != param_count)
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, ")\n");
        }
    }

    fprintf(pyfp, "\n");

    // load weights
    {
        fprintf(pyfp, "        archive = zipfile.ZipFile('%s', 'r')\n", pnnxbinpath.c_str());

        for (const Operator* op : ops)
        {
            if (op->type.substr(0, 3) != "nn." && op->type.substr(0, 16) != "torchvision.ops.")
                continue;

            if (op->type == "nn.quantized.Conv2d" || op->type == "nn.quantized.Linear")
            {
                for (const auto& it : op->attrs)
                {
                    if (it.first == "weight" || it.first == "bias")
                    {
                        fprintf(pyfp, "        self_%s_%s = self.load_pnnx_bin_as_parameter(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), it.first.c_str(), op->name.c_str(), it.first.c_str());
                    }
                    else
                    {
                        // unknown attr
                        continue;
                    }

                    const Attribute& attr = it.second;
                    for (size_t i = 0; i < attr.shape.size(); i++)
                    {
                        fprintf(pyfp, "%d", attr.shape[i]);
                        if (i + 1 != attr.shape.size())
                            fprintf(pyfp, ",");
                    }

                    fprintf(pyfp, "), '%s', requires_grad=False)\n", type_to_numpy_string(attr.type));
                }

                fprintf(pyfp, "        self.%s.set_weight_bias(self_%s_weight, self_%s_bias)\n", sanitize_identifier(op->name).c_str(), sanitize_identifier(op->name).c_str(), sanitize_identifier(op->name).c_str());
                std::string scale_str = float_to_string(op->params.at("scale").f);
                fprintf(pyfp, "        self.%s.scale = %s\n", sanitize_identifier(op->name).c_str(), scale_str.c_str());
                fprintf(pyfp, "        self.%s.zero_point = %d\n", sanitize_identifier(op->name).c_str(), op->params.at("zero_point").i);

                continue;
            }

            for (const auto& it : op->attrs)
            {
                if (it.first == "running_mean" || it.first == "running_var")
                {
                    fprintf(pyfp, "        self.%s.%s = self.load_pnnx_bin_as_tensor(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), it.first.c_str(), op->name.c_str(), it.first.c_str());
                }
                else
                {
                    fprintf(pyfp, "        self.%s.%s = self.load_pnnx_bin_as_parameter(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), it.first.c_str(), op->name.c_str(), it.first.c_str());
                }

                const Attribute& attr = it.second;
                for (size_t i = 0; i < attr.shape.size(); i++)
                {
                    fprintf(pyfp, "%d", attr.shape[i]);
                    if (i + 1 != attr.shape.size())
                        fprintf(pyfp, ",");
                }

                if (attr.type == 1 || attr.type == 2 || attr.type == 3)
                {
                    fprintf(pyfp, "), '%s')\n", type_to_numpy_string(attr.type));
                }
                else
                {
                    fprintf(pyfp, "), '%s', requires_grad=False)\n", type_to_numpy_string(attr.type));
                }
            }
        }

        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Attribute")
                continue;

            const std::string& key = op->attrs.begin()->first;
            const Attribute& attr = op->attrs.begin()->second;

            bool is_running_mean_var = false;
            {
                const Operand* r = op->outputs[0];
                if (r->consumers.size() == 1)
                {
                    const Operator* op2 = r->consumers[0];
                    if (op2->type == "F.batch_norm" || op2->type == "F.instance_norm")
                    {
                        if (r == op2->inputs[1] || r == op2->inputs[2])
                        {
                            is_running_mean_var = true;
                        }
                    }
                }
            }

            bool is_empty = false;
            for (size_t i = 0; i < attr.shape.size(); i++)
            {
                if (attr.shape[i] == 0)
                    is_empty = true;
            }

            if (is_empty)
            {
                fprintf(pyfp, "        self.%s_%s = torch.from_numpy(np.empty((", sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str());

                for (size_t i = 0; i < attr.shape.size(); i++)
                {
                    fprintf(pyfp, "%d,", attr.shape[i]);
                }

                fprintf(pyfp, "), dtype='%s'))\n", type_to_numpy_string(attr.type));
            }
            else
            {
                if (is_running_mean_var)
                {
                    fprintf(pyfp, "        self.%s_%s = self.load_pnnx_bin_as_tensor(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str(), op->name.c_str(), key.c_str());
                }
                else
                {
                    fprintf(pyfp, "        self.%s_%s = self.load_pnnx_bin_as_parameter(archive, '%s.%s', (", sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str(), op->name.c_str(), key.c_str());
                }

                for (size_t i = 0; i < attr.shape.size(); i++)
                {
                    fprintf(pyfp, "%d,", attr.shape[i]);
                }

                if (attr.type == 1 || attr.type == 2 || attr.type == 3)
                {
                    fprintf(pyfp, "), '%s')\n", type_to_numpy_string(attr.type));
                }
                else
                {
                    fprintf(pyfp, "), '%s', requires_grad=False)\n", type_to_numpy_string(attr.type));
                }
            }
        }

        fprintf(pyfp, "        archive.close()\n");
    }

    fprintf(pyfp, "\n");

    // utility function
    {
        fprintf(pyfp, "    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):\n");
        fprintf(pyfp, "        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):\n");
        fprintf(pyfp, "        fd, tmppath = tempfile.mkstemp()\n");
        fprintf(pyfp, "        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:\n");
        fprintf(pyfp, "            tmpf.write(keyfile.read())\n");
        fprintf(pyfp, "        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()\n");
        fprintf(pyfp, "        os.remove(tmppath)\n");
        fprintf(pyfp, "        return torch.from_numpy(m)\n");
    }

    fprintf(pyfp, "\n");

    // def forward
    {
        fprintf(pyfp, "    def forward(self");

        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            fprintf(pyfp, ", v_%s", sanitize_identifier(op->outputs[0]->name).c_str());
        }

        fprintf(pyfp, "):\n");
    }

    // forward body
    {
        for (const Operator* op : ops)
        {
            if (op->type == "pnnx.Input" || op->type == "pnnx.Output")
                continue;

            if (op->type == "pnnx.SliceIndexes")
                continue;

            fprintf(pyfp, "        ");

            if (op->type == "pnnx.Expression")
            {
                // expr
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                std::string expanded_expr = expand_expression(op);
                fprintf(pyfp, " = %s\n", expanded_expr.c_str());
            }
            else if (op->type == "pnnx.Attribute")
            {
                const std::string& key = op->attrs.begin()->first;
                fprintf(pyfp, "v_%s = self.%s_%s\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->name).c_str(), sanitize_identifier(key).c_str());
            }
            else if (op->type == "Tensor.slice")
            {
                // slice expr
                std::string slice_expr = make_slice_expression(op);
                fprintf(pyfp, "v_%s = v_%s[%s]\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), slice_expr.c_str());
            }
            else if (op->type == "Tensor.slice_copy")
            {
                // slice copy expr
                std::string slice_expr = make_slice_expression(op);
                fprintf(pyfp, "v_%s = v_%s\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str());
                fprintf(pyfp, "        v_%s[%s] = v_%s\n", sanitize_identifier(op->outputs[0]->name).c_str(), slice_expr.c_str(), sanitize_identifier(op->inputs[1]->name).c_str());
            }
            else if (op->type == "Tensor.index")
            {
                // index expr
                if (op->inputs.size() == 2)
                {
                    std::string expanded_expr = expand_expression(op->inputs[1]->producer);
                    fprintf(pyfp, "v_%s = v_%s[%s]\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), expanded_expr.c_str());
                }
                else
                {
                    std::string index_expr = make_index_expression(op);
                    fprintf(pyfp, "v_%s = v_%s[%s]\n", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), index_expr.c_str());
                }
            }
            else if (op->type == "Tensor.expand")
            {
                // expand
                fprintf(pyfp, "v_%s = v_%s.%s(", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, "*v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                else
                {
                    const std::vector<int>& sizes = op->params.at("sizes").ai;
                    for (size_t i = 0; i < sizes.size(); i++)
                    {
                        fprintf(pyfp, "%d", sizes[i]);
                        if (i + 1 != sizes.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "Tensor.reshape")
            {
                // reshape
                fprintf(pyfp, "v_%s = v_%s.%s(", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, "*v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                else
                {
                    const std::vector<int>& shape = op->params.at("shape").ai;
                    for (size_t i = 0; i < shape.size(); i++)
                    {
                        if (shape[i] == 0)
                        {
                            // torch does not support numpy style reference
                            fprintf(pyfp, "v_%s.size(%d)", sanitize_identifier(op->inputs[0]->name).c_str(), (int)i);
                        }
                        else
                        {
                            fprintf(pyfp, "%d", shape[i]);
                        }
                        if (i + 1 != shape.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "Tensor.repeat")
            {
                // repeat
                fprintf(pyfp, "v_%s = v_%s.%s(", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, "*v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                else
                {
                    const std::vector<int>& sizes = op->params.at("sizes").ai;
                    for (size_t i = 0; i < sizes.size(); i++)
                    {
                        fprintf(pyfp, "%d", sizes[i]);
                        if (i + 1 != sizes.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "torch.cat" || op->type == "torch.stack")
            {
                // cat
                fprintf(pyfp, "v_%s = %s(", sanitize_identifier(op->outputs[0]->name).c_str(), op->type.c_str());
                if (op->inputs.size() == 1)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
                }
                else
                {
                    fprintf(pyfp, "(");
                    for (size_t i = 0; i < op->inputs.size(); i++)
                    {
                        fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                        if (i + 1 != op->inputs.size())
                            fprintf(pyfp, ", ");
                    }
                    fprintf(pyfp, ")");
                }
                fprintf(pyfp, ", dim=%d", op->params.at("dim").i);
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "torch.einsum")
            {
                // einsum
                fprintf(pyfp, "v_%s = %s(", sanitize_identifier(op->outputs[0]->name).c_str(), op->type.c_str());

                fprintf(pyfp, "\'%s\'", op->params.at("equation").s.c_str());

                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, ", v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "torch.unsqueeze")
            {
                fprintf(pyfp, "v_%s = v_%s", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->inputs[0]->name).c_str());

                if (op->params.at("dim").type == 2)
                {
                    const int dim = op->params.at("dim").i;
                    fprintf(pyfp, ".unsqueeze(%d)", dim);
                }
                else
                {
                    // multiple dims, sort to -1,-2,-3,...,0,1,2,3.... and unroll
                    std::vector<int> dims = op->params.at("dim").ai;
                    std::sort(dims.begin(), dims.end(), [](int a, int b) {
                        if (a < 0 && b >= 0) return true;
                        if (a >= 0 && b < 0) return false;
                        if (a < 0 && b < 0) return a > b;
                        return a < b;
                    });
                    for (size_t i = 0; i < dims.size(); i++)
                    {
                        int dim = dims[i];
                        fprintf(pyfp, ".unsqueeze(%d)", dim);
                    }
                }

                fprintf(pyfp, "\n");
            }
            else if (op->type == "torch.prod")
            {
                fprintf(pyfp, "v_%s = ", sanitize_identifier(op->outputs[0]->name).c_str());

                if (op->params.at("dim").type == 2)
                {
                    const int dim = op->params.at("dim").i;
                    const bool keepdim = op->params.at("keepdim").b;
                    fprintf(pyfp, "torch.prod(input=v_%s, dim=%d, keepdim=%s)", sanitize_identifier(op->inputs[0]->name).c_str(), dim, keepdim ? "True" : "False");
                }
                else
                {
                    // multiple dims, sort to ...,-3,-2,-1,...,3,2,1,0 and unroll
                    std::vector<int> dims = op->params.at("dim").ai;
                    const bool keepdim = op->params.at("keepdim").b;
                    std::sort(dims.begin(), dims.end(), [](int a, int b) {
                        if (a < 0 && b >= 0) return true;
                        if (a >= 0 && b < 0) return false;
                        if (a < 0 && b < 0) return a < b;
                        return a > b;
                    });
                    for (size_t i = 0; i < dims.size(); i++)
                    {
                        fprintf(pyfp, "torch.prod(input=");
                    }
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
                    for (size_t i = 0; i < dims.size(); i++)
                    {
                        int dim = dims[i];
                        fprintf(pyfp, ", dim=%d, keepdim=%s)", dim, keepdim ? "True" : "False");
                    }
                }

                fprintf(pyfp, "\n");
            }
            else if (op->type == "prim::TupleUnpack")
            {
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, " = v_%s\n", sanitize_identifier(op->inputs[0]->name).c_str());
            }
            else if (op->type == "prim::TupleConstruct")
            {
                fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[0]->name).c_str());
                fprintf(pyfp, " = (");
                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s, ", sanitize_identifier(op->inputs[i]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "prim::ListUnpack")
            {
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, " = v_%s\n", sanitize_identifier(op->inputs[0]->name).c_str());
            }
            else if (op->type == "prim::ListConstruct")
            {
                fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[0]->name).c_str());
                fprintf(pyfp, " = [");
                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                    if (i + 1 != op->inputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "]\n");
            }
            else if (op->type == "nn.GRU" || op->type == "nn.RNN")
            {
                if (op->outputs.size() == 1)
                {
                    fprintf(pyfp, "v_%s, _", sanitize_identifier(op->outputs[0]->name).c_str());
                }
                else
                {
                    fprintf(pyfp, "v_%s, v_%s", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->outputs[1]->name).c_str());
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
                if (op->inputs.size() == 2)
                {
                    fprintf(pyfp, ", v_%s", sanitize_identifier(op->inputs[1]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "nn.LSTM")
            {
                if (op->outputs.size() == 1)
                {
                    fprintf(pyfp, "v_%s, _", sanitize_identifier(op->outputs[0]->name).c_str());
                }
                else
                {
                    fprintf(pyfp, "v_%s, (v_%s, v_%s)", sanitize_identifier(op->outputs[0]->name).c_str(), sanitize_identifier(op->outputs[1]->name).c_str(), sanitize_identifier(op->outputs[2]->name).c_str());
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
                if (op->inputs.size() == 3)
                {
                    fprintf(pyfp, ", (v_%s, v_%s)", sanitize_identifier(op->inputs[1]->name).c_str(), sanitize_identifier(op->inputs[2]->name).c_str());
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type == "nn.MultiheadAttention")
            {
                bool need_weights = true;
                if (op->outputs.size() == 1)
                {
                    fprintf(pyfp, "v_%s, _", sanitize_identifier(op->outputs[0]->name).c_str());
                    need_weights = false;
                }
                else
                {
                    for (size_t i = 0; i < op->outputs.size(); i++)
                    {
                        fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                        if (i + 1 != op->outputs.size())
                            fprintf(pyfp, ", ");
                    }
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                if (op->inputs.size() == 1)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    fprintf(pyfp, "v_%s, v_%s, v_%s", in0.c_str(), in0.c_str(), in0.c_str());
                }
                else if (op->inputs.size() == 2)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    std::string in1 = sanitize_identifier(op->inputs[1]->name);
                    if (op->inputnames.size() == 2 && op->inputnames[1] == "attn_mask")
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, attn_mask=v_%s", in0.c_str(), in0.c_str(), in0.c_str(), in1.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s", in0.c_str(), in1.c_str(), in1.c_str());
                    }
                }
                else if (op->inputs.size() == 3)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    std::string in1 = sanitize_identifier(op->inputs[1]->name);
                    std::string in2 = sanitize_identifier(op->inputs[2]->name);
                    if (op->inputnames.size() == 3 && op->inputnames[2] == "attn_mask")
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, attn_mask=v_%s", in0.c_str(), in1.c_str(), in1.c_str(), in2.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s", in0.c_str(), in1.c_str(), in2.c_str());
                    }
                }
                else if (op->inputs.size() == 4)
                {
                    std::string in0 = sanitize_identifier(op->inputs[0]->name);
                    std::string in1 = sanitize_identifier(op->inputs[1]->name);
                    std::string in2 = sanitize_identifier(op->inputs[2]->name);
                    std::string in3 = sanitize_identifier(op->inputs[3]->name);
                    if (op->inputnames.size() == 4 && op->inputnames[3] == "attn_mask")
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, attn_mask=v_%s", in0.c_str(), in1.c_str(), in2.c_str(), in3.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, "v_%s, v_%s, v_%s, v_%s", in0.c_str(), in1.c_str(), in2.c_str(), in3.c_str());
                    }
                }
                else
                {
                    for (size_t i = 0; i < op->inputs.size(); i++)
                    {
                        fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                        if (i + 1 != op->inputs.size())
                            fprintf(pyfp, ", ");
                    }
                }
                if (need_weights)
                {
                    fprintf(pyfp, ", need_weights=True");
                }
                else
                {
                    fprintf(pyfp, ", need_weights=False");
                }
                fprintf(pyfp, ")\n");
            }
            else if (op->type.substr(0, 3) == "nn." || op->type.substr(0, 16) == "torchvision.ops.")
            {
                // self.xxx()
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, " = self.%s(", sanitize_identifier(op->name).c_str());
                for (size_t i = 0; i < op->inputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                    if (i + 1 != op->inputs.size())
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, ")\n");
            }
            else
            {
                if (op->type.find("::") == std::string::npos && op->type.find(".") == std::string::npos)
                {
                    fprintf(stderr, "todo %s\n", op->type.c_str());
                }

                // direct
                for (size_t i = 0; i < op->outputs.size(); i++)
                {
                    fprintf(pyfp, "v_%s", sanitize_identifier(op->outputs[i]->name).c_str());
                    if (i + 1 != op->outputs.size())
                        fprintf(pyfp, ", ");
                }

                if (op->type == "torch.max" || op->type == "torch.min")
                {
                    if (op->has_param("dim") && op->outputs.size() == 1)
                    {
                        // torch.max and torch.min with dim returns tuple
                        fprintf(pyfp, ", _");
                    }
                }

                if (op->type.substr(0, 7) == "Tensor.")
                {
                    if (op->type == "Tensor.fill")
                    {
                        fprintf(pyfp, " = v_%s.fill_(", sanitize_identifier(op->inputs[0]->name).c_str());
                    }
                    else
                    {
                        fprintf(pyfp, " = v_%s.%s(", sanitize_identifier(op->inputs[0]->name).c_str(), op->type.substr(7).c_str());
                    }

                    if (op->inputnames.size() == op->inputs.size())
                    {
                        for (size_t i = 1; i < op->inputs.size(); i++)
                        {
                            if (!op->inputnames[i].empty())
                                continue;

                            fprintf(pyfp, "v_%s, ", sanitize_identifier(op->inputs[i]->name).c_str());
                        }

                        for (size_t i = 1; i < op->inputs.size(); i++)
                        {
                            if (op->inputnames[i].empty())
                                continue;

                            fprintf(pyfp, "%s=v_%s, ", op->inputnames[i].c_str(), sanitize_identifier(op->inputs[i]->name).c_str());
                        }
                    }
                    else
                    {
                        for (size_t i = 1; i < op->inputs.size(); i++)
                        {
                            fprintf(pyfp, "v_%s, ", sanitize_identifier(op->inputs[i]->name).c_str());
                        }
                    }
                }
                else
                {
                    fprintf(pyfp, " = %s(", op->type.c_str());

                    if (op->inputnames.size() == op->inputs.size())
                    {
                        for (size_t i = 0; i < op->inputs.size(); i++)
                        {
                            bool is_input = i == 0 && op->inputnames[0] == "input";
                            if (!op->inputnames[i].empty() && !is_input)
                                continue;

                            fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                            if (i + 1 != op->inputs.size())
                                fprintf(pyfp, ", ");
                        }

                        for (size_t i = 0; i < op->inputs.size(); i++)
                        {
                            bool is_input = i == 0 && op->inputnames[0] == "input";
                            if (op->inputnames[i].empty() || is_input)
                                continue;

                            fprintf(pyfp, "%s=v_%s", op->inputnames[i].c_str(), sanitize_identifier(op->inputs[i]->name).c_str());
                            if (i + 1 != op->inputs.size())
                                fprintf(pyfp, ", ");
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < op->inputs.size(); i++)
                        {
                            fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[i]->name).c_str());
                            if (i + 1 != op->inputs.size())
                                fprintf(pyfp, ", ");
                        }
                    }
                }

                int i = 0;
                for (const auto& it : op->params)
                {
                    if (op->type.substr(0, 7) == "Tensor." && i == 0)
                    {
                        fprintf(pyfp, "%s=", it.first.c_str());
                    }
                    else if (op->type == "F.pad" && op->params.at("mode").s != "constant" && it.first == "value")
                    {
                        // skip F.pad value for non constant pad mode
                        i++;
                        continue;
                    }
                    else if (op->inputs.empty() && i == 0)
                    {
                        fprintf(pyfp, "%s=", it.first.c_str());
                    }
                    else
                    {
                        fprintf(pyfp, ", %s=", it.first.c_str());
                    }

                    i++;

                    bool scalar_as_tensor = false;
                    if ((op->type == "Tensor.index_put" && it.first == "values")
                            || (op->type == "torch.where" && it.first == "input")
                            || (op->type == "torch.where" && it.first == "other"))
                    {
                        scalar_as_tensor = true;
                    }

                    const Parameter& param = it.second;
                    if (param.type == 0)
                    {
                        if (scalar_as_tensor)
                        {
                            fprintf(pyfp, "torch.tensor(False)");
                        }
                        else
                        {
                            fprintf(pyfp, "None");
                        }
                    }
                    if (param.type == 1)
                    {
                        if (param.b)
                            fprintf(pyfp, "True");
                        else
                            fprintf(pyfp, "False");
                    }
                    if (param.type == 2)
                    {
                        if (scalar_as_tensor)
                        {
                            fprintf(pyfp, "torch.tensor(%d)", param.i);
                        }
                        else
                        {
                            fprintf(pyfp, "%d", param.i);
                        }
                    }
                    if (param.type == 3)
                    {
                        if (scalar_as_tensor)
                        {
                            if (param.f == (int)param.f)
                                fprintf(pyfp, "torch.tensor(%.1f)", param.f);
                            else
                                fprintf(pyfp, "torch.tensor(%g)", param.f);
                        }
                        else
                        {
                            if (param.f == (int)param.f)
                                fprintf(pyfp, "%.1f", param.f);
                            else
                                fprintf(pyfp, "%g", param.f);
                        }
                    }
                    if (param.type == 4)
                    {
                        if (param.s.substr(0, 6) == "torch.")
                        {
                            fprintf(pyfp, "%s", param.s.c_str());
                        }
                        else if (scalar_as_tensor)
                        {
                            if (param.s == "inf" || param.s == "-inf")
                            {
                                fprintf(pyfp, "torch.tensor(float(\'%s\'))", param.s.c_str());
                            }
                            else
                            {
                                fprintf(pyfp, "torch.tensor(\'%s\')", param.s.c_str());
                            }
                        }
                        else
                        {
                            if (param.s == "inf" || param.s == "-inf")
                            {
                                fprintf(pyfp, "float(\'%s\')", param.s.c_str());
                            }
                            else
                            {
                                fprintf(pyfp, "\'%s\'", param.s.c_str());
                            }
                        }
                    }
                    if (param.type == 5)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.ai.size(); i++)
                        {
                            if ((op->type == "F.adaptive_avg_pool2d"
                                    || op->type == "F.adaptive_avg_pool3d"
                                    || op->type == "F.adaptive_max_pool2d"
                                    || op->type == "F.adaptive_max_pool3d")
                                    && it.first == "output_size" && param.ai[i] == 0)
                            {
                                fprintf(pyfp, "None");
                            }
                            else
                            {
                                fprintf(pyfp, "%d", param.ai[i]);
                            }
                            if (i + 1 != param.ai.size() || param.ai.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                    if (param.type == 6)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.af.size(); i++)
                        {
                            std::string afs = float_to_string(param.af[i]);
                            fprintf(pyfp, "%s", afs.c_str());
                            if (i + 1 != param.af.size() || param.af.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                    if (param.type == 7)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.as.size(); i++)
                        {
                            if (param.as[i].substr(0, 6) == "torch.")
                            {
                                fprintf(pyfp, "%s", param.as[i].c_str());
                            }
                            else
                            {
                                fprintf(pyfp, "\'%s\'", param.as[i].c_str());
                            }
                            if (i + 1 != param.as.size() || param.as.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                    if (param.type == 10)
                    {
                        fprintf(pyfp, "(%f%+fj)", param.c.real(), param.c.imag());
                    }
                    if (param.type == 11)
                    {
                        fprintf(pyfp, "(");
                        for (size_t i = 0; i < param.ac.size(); i++)
                        {
                            fprintf(pyfp, "(%f%+fj)", param.ac[i].real(), param.ac[i].imag());
                            if (i + 1 != param.ac.size() || param.ac.size() == 1)
                                fprintf(pyfp, ",");
                        }
                        fprintf(pyfp, ")");
                    }
                }

                fprintf(pyfp, ")\n");
            }
        }
    }

    // return
    {
        fprintf(pyfp, "        return ");

        int output_count = 0;
        {
            for (const Operator* op : ops)
            {
                if (op->type == "pnnx.Output")
                    output_count++;
            }
        }

        int output_index = 0;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Output")
                continue;

            fprintf(pyfp, "v_%s", sanitize_identifier(op->inputs[0]->name).c_str());
            if (output_index + 1 != output_count)
                fprintf(pyfp, ", ");

            output_index++;
        }

        fprintf(pyfp, "\n");
    }

    fprintf(pyfp, "\n");

    // export torchscript
    {
        fprintf(pyfp, "def export_torchscript():\n");
        fprintf(pyfp, "    net = Model()\n");
        fprintf(pyfp, "    net.float()\n");
        fprintf(pyfp, "    net.eval()\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    torch.manual_seed(0)\n");

        std::vector<std::string> input_names;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            const Operand* r = op->outputs[0];
            std::string input_name = std::string("v_") + sanitize_identifier(r->name);
            if (type_is_integer(r->type))
            {
                fprintf(pyfp, "    %s = torch.randint(10, (", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d", r->shape[i]);
                    if (i + 1 != r->shape.size() || r->shape.size() == 1)
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "), dtype=%s)\n", type_to_dtype_string(r->type));
            }
            else
            {
                fprintf(pyfp, "    %s = torch.rand(", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d, ", r->shape[i]);
                }
                fprintf(pyfp, "dtype=%s)\n", type_to_dtype_string(r->type));
            }

            input_names.push_back(input_name);
        }

        fprintf(pyfp, "\n");

        if (input_names.size() == 1)
        {
            fprintf(pyfp, "    mod = torch.jit.trace(net, %s)\n", input_names[0].c_str());
        }
        else
        {
            fprintf(pyfp, "    mod = torch.jit.trace(net, (");

            for (size_t i = 0; i < input_names.size(); i++)
            {
                fprintf(pyfp, "%s", input_names[i].c_str());
                if (i + 1 != input_names.size())
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, "))\n");
        }

        fprintf(pyfp, "    mod.save(\"%s.pt\")\n", pypath.c_str());
    }

    fprintf(pyfp, "\n");

    // export onnx
    {
        fprintf(pyfp, "def export_onnx():\n");
        fprintf(pyfp, "    net = Model()\n");
        fprintf(pyfp, "    net.float()\n");
        fprintf(pyfp, "    net.eval()\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    torch.manual_seed(0)\n");

        std::vector<std::string> input_names;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            const Operand* r = op->outputs[0];
            std::string input_name = std::string("v_") + sanitize_identifier(r->name);
            if (type_is_integer(r->type))
            {
                fprintf(pyfp, "    %s = torch.randint(10, (", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d", r->shape[i]);
                    if (i + 1 != r->shape.size() || r->shape.size() == 1)
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "), dtype=%s)\n", type_to_dtype_string(r->type));
            }
            else
            {
                fprintf(pyfp, "    %s = torch.rand(", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d, ", r->shape[i]);
                }
                fprintf(pyfp, "dtype=%s)\n", type_to_dtype_string(r->type));
            }

            input_names.push_back(input_name);
        }

        fprintf(pyfp, "\n");

        // torch.onnx.export(net, v_0, "test_swin_t.onnx", export_params=True, opset_version=14, input_names=['in0'], output_names=['out0'])

        if (input_names.size() == 1)
        {
            fprintf(pyfp, "    torch.onnx.export(net, %s", input_names[0].c_str());
        }
        else
        {
            fprintf(pyfp, "    torch.onnx.export(net, (");

            for (size_t i = 0; i < input_names.size(); i++)
            {
                fprintf(pyfp, "%s", input_names[i].c_str());
                if (i + 1 != input_names.size())
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, ")");
        }

        fprintf(pyfp, ", \"%s.onnx\", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13", pypath.c_str());

        fprintf(pyfp, ", input_names=[");
        {
            int input_count = 0;
            {
                for (const Operator* op : ops)
                {
                    if (op->type == "pnnx.Input")
                        input_count++;
                }
            }

            int input_index = 0;
            for (const Operator* op : ops)
            {
                if (op->type != "pnnx.Input")
                    continue;

                fprintf(pyfp, "'in%d'", input_index);
                if (input_index + 1 != input_count)
                    fprintf(pyfp, ", ");

                input_index++;
            }
        }
        fprintf(pyfp, "]");

        fprintf(pyfp, ", output_names=[");
        {
            int output_count = 0;
            {
                for (const Operator* op : ops)
                {
                    if (op->type == "pnnx.Output")
                        output_count++;
                }
            }

            int output_index = 0;
            for (const Operator* op : ops)
            {
                if (op->type != "pnnx.Output")
                    continue;

                fprintf(pyfp, "'out%d'", output_index);
                if (output_index + 1 != output_count)
                    fprintf(pyfp, ", ");

                output_index++;
            }
        }
        fprintf(pyfp, "]");

        fprintf(pyfp, ")\n");
    }

    fprintf(pyfp, "\n");

    // export pnnx
    {
        fprintf(pyfp, "def export_pnnx():\n");
        fprintf(pyfp, "    net = Model()\n");
        fprintf(pyfp, "    net.float()\n");
        fprintf(pyfp, "    net.eval()\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    torch.manual_seed(0)\n");

        std::vector<std::string> input_names;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            const Operand* r = op->outputs[0];
            std::string input_name = std::string("v_") + sanitize_identifier(r->name);
            if (type_is_integer(r->type))
            {
                fprintf(pyfp, "    %s = torch.randint(10, (", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d", r->shape[i]);
                    if (i + 1 != r->shape.size() || r->shape.size() == 1)
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "), dtype=%s)\n", type_to_dtype_string(r->type));
            }
            else
            {
                fprintf(pyfp, "    %s = torch.rand(", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d, ", r->shape[i]);
                }
                fprintf(pyfp, "dtype=%s)\n", type_to_dtype_string(r->type));
            }

            input_names.push_back(input_name);
        }

        fprintf(pyfp, "\n");

        fprintf(pyfp, "    import pnnx\n");
        if (input_names.size() == 1)
        {
            fprintf(pyfp, "    pnnx.export(net, \"%s.pt\", %s)\n", pypath.c_str(), input_names[0].c_str());
        }
        else
        {
            fprintf(pyfp, "    pnnx.export(net, \"%s.pt\", (", pypath.c_str());

            for (size_t i = 0; i < input_names.size(); i++)
            {
                fprintf(pyfp, "%s", input_names[i].c_str());
                if (i + 1 != input_names.size())
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, "))\n");
        }
    }

    fprintf(pyfp, "\n");

    // export ncnn
    {
        fprintf(pyfp, "def export_ncnn():\n");
        fprintf(pyfp, "    export_pnnx()\n");
    }

    fprintf(pyfp, "\n");

    // test inference
    {
        fprintf(pyfp, "@torch.no_grad()\n");
        fprintf(pyfp, "def test_inference():\n");
        fprintf(pyfp, "    net = Model()\n");
        fprintf(pyfp, "    net.float()\n");
        fprintf(pyfp, "    net.eval()\n");
        fprintf(pyfp, "\n");
        fprintf(pyfp, "    torch.manual_seed(0)\n");

        int input_shapes_i = 0;

        std::vector<std::string> input_names;
        for (const Operator* op : ops)
        {
            if (op->type != "pnnx.Input")
                continue;

            const Operand* r = op->outputs[0];

            std::vector<int> input_shape;
            if (input_shapes.empty())
            {
                input_shape = r->shape;
            }
            else
            {
                const std::vector<int64_t>& s = input_shapes[input_shapes_i++];
                for (int64_t d : s)
                {
                    input_shape.push_back((int)d);
                }
            }

            std::string input_name = std::string("v_") + sanitize_identifier(r->name);
            if (type_is_integer(r->type))
            {
                fprintf(pyfp, "    %s = torch.randint(10, (", input_name.c_str());
                for (size_t i = 0; i < input_shape.size(); i++)
                {
                    int dimsize = input_shape[i];
                    if (dimsize == -1)
                        dimsize = 128; // try with a good default
                    fprintf(pyfp, "%d", dimsize);
                    if (i + 1 != input_shape.size() || input_shape.size() == 1)
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "), dtype=%s)\n", type_to_dtype_string(r->type));
            }
            else
            {
                fprintf(pyfp, "    %s = torch.rand(", input_name.c_str());
                for (size_t i = 0; i < input_shape.size(); i++)
                {
                    int dimsize = input_shape[i];
                    if (dimsize == -1)
                        dimsize = 128; // try with a good default
                    fprintf(pyfp, "%d, ", dimsize);
                }
                fprintf(pyfp, "dtype=%s)\n", type_to_dtype_string(r->type));
            }

            input_names.push_back(input_name);
        }

        fprintf(pyfp, "\n");

        if (input_names.size() == 1)
        {
            fprintf(pyfp, "    return net(%s)\n", input_names[0].c_str());
        }
        else
        {
            fprintf(pyfp, "    return net(");

            for (size_t i = 0; i < input_names.size(); i++)
            {
                fprintf(pyfp, "%s", input_names[i].c_str());
                if (i + 1 != input_names.size())
                    fprintf(pyfp, ", ");
            }

            fprintf(pyfp, ")\n");
        }
    }

    fprintf(pyfp, "\n");

    // main
    {
        fprintf(pyfp, "if __name__ == \"__main__\":\n");
        fprintf(pyfp, "    print(test_inference())\n");
    }

    fclose(pyfp);

    return 0;
}

int Graph::parse(const std::string& param)
{
    std::istringstream is(param);
    if (!is.good())
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    int magic = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> magic;
    }

    int operator_count = 0;
    int operand_count = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> operator_count >> operand_count;
    }

    for (int i = 0; i < operator_count; i++)
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        std::string type;
        std::string name;
        int input_count = 0;
        int output_count = 0;

        iss >> type >> name >> input_count >> output_count;

        Operator* op = new_operator(type, name);

        for (int j = 0; j < input_count; j++)
        {
            std::string operand_name;
            iss >> operand_name;

            Operand* r = get_operand(operand_name);
            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }

        for (int j = 0; j < output_count; j++)
        {
            std::string operand_name;
            iss >> operand_name;

            Operand* r = new_operand(operand_name);
            r->producer = op;
            op->outputs.push_back(r);
        }

        // key=value
        while (!iss.eof())
        {
            std::string param;
            iss >> param;

            std::string key;
            std::string value;
            std::istringstream pss(param);
            std::getline(pss, key, '=');
            std::getline(pss, value);

            if (key[0] == '@')
            {
                // attribute
                //                 load_attribute(op, key.substr(1), value, szr);
                op->attrs[key.substr(1)] = Attribute();

                Attribute& attr = op->attrs[key.substr(1)];

                attr.type = 0;
                if (value.empty())
                    continue;

                if (value[0] == '%')
                {
                    // @data=%op1.data
                    attr.data = std::vector<char>(value.begin(), value.end());
                }

                if (value[0] == '(')
                {
                    // @data=(1,%c,?,4)f32

                    // type
                    std::string typestr = value.substr(value.find_last_of(')') + 1);
                    attr.type = string_to_type(typestr.c_str());

                    // shape
                    std::string lc = value.substr(1, value.find_last_of(')') - 1);
                    std::istringstream lcss(lc);

                    attr.shape.clear();
                    while (!lcss.eof())
                    {
                        std::string elem;
                        std::getline(lcss, elem, ',');

                        if (elem == "?")
                        {
                            attr.shape.push_back(-1);
                        }
                        else if (elem[0] == '%')
                        {
                            // encode %abc as symbolic tag
                            attr.shape.push_back(-233);
                            int index = attr.shape.size() - 1;
                            std::string key = elem.substr(1);
                            attr.params[std::string("__shape__") + std::to_string(index)] = key;
                        }
                        else
                        {
                            int i = std::stoi(elem);
                            attr.shape.push_back(i);
                        }
                    }
                }
            }
            else if (key[0] == '$')
            {
                // operand input key
                load_input_key(op, key.substr(1), value);
            }
            else if (key[0] == '#')
            {
                // operand shape
                load_shape(op, key.substr(1), value);
            }
            else
            {
                // parameter
                load_parameter(op, key, value);
            }
        }
    }

    return 0;
}

void Operand::remove_consumer(const Operator* c)
{
    auto it = std::find(consumers.begin(), consumers.end(), c);
    if (it != consumers.end())
        consumers.erase(it);
}

Operator* Graph::new_operator(const std::string& type, const std::string& name)
{
    Operator* op = new Operator;
    op->type = type;
    op->name = name;
    ops.push_back(op);
    return op;
}

Operator* Graph::new_operator_before(const std::string& type, const std::string& name, const Operator* cur)
{
    Operator* op = new Operator;
    op->type = type;
    op->name = name;
    ops.insert(std::find(ops.begin(), ops.end(), cur), op);
    return op;
}

Operator* Graph::new_operator_after(const std::string& type, const std::string& name, const Operator* cur)
{
    Operator* op = new Operator;
    op->type = type;
    op->name = name;
    ops.insert(std::find(ops.begin(), ops.end(), cur) + 1, op);
    return op;
}

Operand* Graph::new_operand(const std::string& name)
{
    Operand* r = new Operand;
    r->name = name;
    operands.push_back(r);
    return r;
}

Operand* Graph::get_operand(const std::string& name)
{
    for (Operand* r : operands)
    {
        if (r->name == name)
            return r;
    }

    return 0;
}

const Operand* Graph::get_operand(const std::string& name) const
{
    for (const Operand* r : operands)
    {
        if (r->name == name)
            return r;
    }

    return 0;
}

pnnx::ModelInfo Graph::flops_mem_count()
{
    for (const Operator* op : ops)
    {
        if (op->type == "nn.Conv1d" || op->type == "nn.ConvTranspose1d" && op->inputs[0]->shape.size() == 3)
        {
            if (op->inputs[0]->type != 0)
            {
                int in_n, in_c, in_l, out_c, out_l, k_s, g;
                bool bias;
                in_n = op->inputs[0]->shape[0];
                in_c = op->inputs[0]->shape[1];
                in_l = op->inputs[0]->shape[2];
                k_s = op->params.at("kernel_size").i;
                out_c = op->params.at("out_channels").i;
                out_l = op->outputs[0]->shape[2];
                bias = op->params.at("bias").b; //bias
                if (bias)
                {
                    m.flops += in_n * 2 * in_c * k_s * out_c * out_l;
                    m.memory_access += in_n * (in_l * in_c + out_l * out_c + in_c * k_s * out_c + out_c);
                }
                else
                {
                    m.flops += in_n * (2 * in_c * k_s - 1) * out_c * out_l;
                    m.memory_access += in_n * (in_l * in_c + out_l * out_c + in_c * k_s * out_c);
                }
            }
        }
        else if (op->type == "F.conv1d" || op->type == "F.conv_transpose1d" && op->inputs[0]->shape.size() == 3)
        {
            if (op->inputs[0]->type != 0)
            {
                int in_n, in_c, in_l, out_c, out_l, k_s, g;
                bool bias = true;
                in_n = op->inputs[0]->shape[0];
                in_c = op->inputs[0]->shape[1];
                in_l = op->inputs[0]->shape[2];
                k_s = op->inputs[1]->shape[2];
                out_c = op->outputs[0]->shape[1];
                out_l = op->outputs[0]->shape[2];
                if (op->params.find("bias") != op->params.end())
                {
                    std::string val = Parameter::encode_to_string(op->params.at("bias"));
                    if (val == "None")
                    {
                        bias = false;
                    }
                }
                if (bias)
                {
                    m.flops += in_n * 2 * in_c * k_s * out_c * out_l;
                    m.memory_access += in_n * (in_l * in_c + out_l * out_c + in_c * k_s * out_c + out_c);
                }
                else
                {
                    m.flops += in_n * (2 * in_c * k_s - 1) * out_c * out_l;
                    m.memory_access += in_n * (in_l * in_c + out_l * out_c + in_c * k_s * out_c);
                }
            }
        }
        else if (op->type == "nn.Conv2d" || op->type == "nn.ConvTranspose2d" && op->inputs[0]->shape.size() == 4)
        {
            if (op->inputs[0]->type != 0)
            {
                int in_n, in_c, in_h, in_w, out_c, out_h, out_w, k_h, k_w, g;
                bool bias;
                in_n = op->inputs[0]->shape[0];
                in_c = op->inputs[0]->shape[1];
                in_h = op->inputs[0]->shape[2];
                in_w = op->inputs[0]->shape[3];
                k_h = op->params.at("kernel_size").ai[0];
                k_w = op->params.at("kernel_size").ai[1];
                out_c = op->params.at("out_channels").i;
                out_h = op->outputs[0]->shape[2];
                out_w = op->outputs[0]->shape[3];
                bias = op->params.at("bias").b; //bias
                if (bias)
                {
                    m.flops += in_n * 2 * in_c * k_h * k_w * out_c * out_w * out_h;
                    m.memory_access += in_n * (in_w * in_h * in_c + out_w * out_h * out_c + in_c * k_h * k_w * out_c + out_c);
                }
                else
                {
                    m.flops += in_n * (2 * in_c * k_h * k_w - 1) * out_c * out_w * out_h;
                    m.memory_access += in_n * (in_w * in_h * in_c + out_w * out_h * out_c + k_h * k_w * in_c * out_c);
                }
            }
        }
        else if (op->type == "F.conv2d" || op->type == "F.conv_transpose2d" && op->inputs[0]->shape.size() == 4)
        {
            if (op->inputs[0]->type != 0)
            {
                int in_n, in_c, in_h, in_w, out_c, out_h, out_w, k_h, k_w, g;
                bool bias = true;
                in_n = op->inputs[0]->shape[0];
                in_c = op->inputs[0]->shape[1];
                in_h = op->inputs[0]->shape[2];
                in_w = op->inputs[0]->shape[3];
                k_h = op->inputs[1]->shape[2];
                k_w = op->inputs[1]->shape[3];
                out_c = op->outputs[0]->shape[1];
                out_h = op->outputs[0]->shape[2];
                out_w = op->outputs[0]->shape[3];
                if (op->params.find("bias") != op->params.end())
                {
                    std::string val = Parameter::encode_to_string(op->params.at("bias"));
                    if (val == "None")
                    {
                        bias = false;
                    }
                }
                if (bias)
                {
                    m.flops += in_n * 2 * in_c * k_h * k_w * out_c * out_w * out_h;
                    m.memory_access += in_n * (in_w * in_h * in_c + out_w * out_h * out_c + in_c * k_h * k_w * out_c + out_c);
                }
                else
                {
                    m.flops += in_n * (2 * in_c * k_h * k_w - 1) * out_c * out_w * out_h;
                    m.memory_access += in_n * (in_w * in_h * in_c + out_w * out_h * out_c + k_h * k_w * in_c * out_c);
                }
            }
        }
        else if (op->type == "nn.Conv3d" || op->type == "nn.ConvTranspose3d" && op->inputs[0]->shape.size() == 5)
        {
            if (op->inputs[0]->type != 0)
            {
                int in_n, in_c, in_d, in_h, in_w, out_c, out_d, out_h, out_w, k_d, k_h, k_w, g;
                bool bias;
                in_n = op->inputs[0]->shape[0];
                in_c = op->inputs[0]->shape[1];
                in_d = op->inputs[0]->shape[2];
                in_h = op->inputs[0]->shape[3];
                in_w = op->inputs[0]->shape[4];
                k_d = op->params.at("kernel_size").ai[0];
                k_h = op->params.at("kernel_size").ai[1];
                k_w = op->params.at("kernel_size").ai[2];
                out_c = op->outputs[0]->shape[1];
                out_d = op->outputs[0]->shape[2];
                out_h = op->outputs[0]->shape[3];
                out_w = op->outputs[0]->shape[4];
                bias = op->params.at("bias").b; //bias
                if (bias)
                {
                    m.flops += in_n * 2 * in_c * k_d * k_h * k_w * out_c * out_d * out_w * out_h;
                    m.memory_access += in_n * (in_d * in_w * in_h * in_c + out_d * out_w * out_h * out_c + in_c * k_d * k_h * k_w * out_c + out_c);
                }
                else
                {
                    m.flops += in_n * (2 * in_c * k_d * k_h * k_w - 1) * out_c * out_d * out_w * out_h;
                    m.memory_access += in_n * (in_d * in_w * in_h * in_c + out_d * out_w * out_h * out_c + in_c * k_d * k_h * k_w * out_c);
                }
            }
        }
        else if (op->type == "F.conv3d" || op->type == "F.conv_transpose3d" && op->inputs[0]->shape.size() == 5)
        {
            if (op->inputs[0]->type != 0)
            {
                int in_n, in_c, in_d, in_h, in_w, out_c, out_d, out_h, out_w, k_d, k_h, k_w, g;
                bool bias = true;
                in_n = op->inputs[0]->shape[0];
                in_c = op->inputs[0]->shape[1];
                in_d = op->inputs[0]->shape[2];
                in_h = op->inputs[0]->shape[3];
                in_w = op->inputs[0]->shape[4];
                k_d = op->inputs[1]->shape[2];
                k_h = op->inputs[1]->shape[3];
                k_w = op->inputs[1]->shape[4];
                out_c = op->outputs[0]->shape[1];
                out_d = op->outputs[0]->shape[2];
                out_h = op->outputs[0]->shape[3];
                out_w = op->outputs[0]->shape[4];
                if (op->params.find("bias") != op->params.end())
                {
                    std::string val = Parameter::encode_to_string(op->params.at("bias"));
                    if (val == "None")
                    {
                        bias = false;
                    }
                }
                if (bias)
                {
                    m.flops += in_n * 2 * in_c * k_d * k_h * k_w * out_c * out_d * out_w * out_h;
                    m.memory_access += in_n * (in_d * in_w * in_h * in_c + out_d * out_w * out_h * out_c + in_c * k_d * k_h * k_w * out_c + out_c);
                }
                else
                {
                    m.flops += in_n * (2 * in_c * k_d * k_h * k_w - 1) * out_c * out_d * out_w * out_h;
                    m.memory_access += in_n * (in_d * in_w * in_h * in_c + out_d * out_w * out_h * out_c + in_c * k_d * k_h * k_w * out_c);
                }
            }
        }
        else if (op->type == "nn.Linear" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, in, out, mem = 1;
            bool bias;
            in = op->params.at("in_features").i;
            out = op->params.at("out_features").i;
            bias = op->params.at("bias").b;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size - 1; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            if (bias)
            {
                m.flops += mem * 2 * in * out;
                m.memory_access += mem * (in + out + in * out + 1);
            }
            else
            {
                m.flops += mem * (2 * in - 1) * out;
                m.memory_access += mem * (in + out + in * out);
            }
        }
        else if (op->type == "F.linear" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, in, out, mem = 1;
            bool bias = true;
            in = op->inputs[1]->shape[1];
            out = op->inputs[1]->shape[0];
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size - 1; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            if (op->params.find("bias") != op->params.end())
            {
                std::string val = Parameter::encode_to_string(op->params.at("bias"));
                if (val == "None")
                {
                    bias = false;
                }
            }
            if (bias)
            {
                m.flops += mem * 2 * in * out;
                m.memory_access += mem * (in + out + in * out + 1);
            }
            else
            {
                m.flops += mem * (2 * in - 1) * out;
                m.memory_access += mem * (in + out + in * out);
            }
        }
        else if (op->type == "nn.MultiheadAttention" && op->inputs[0]->shape.size() == 3)
        {
            int in_size, q_l, k_s, v_s, num_heads, embed_dim, Kdim, vdim;
            long long linear1, attention, linerar2, weights, in, attention_m, out;
            bool batch_first = op->params.find("batch_first") != op->params.end() && op->params.at("batch_first").b;
            in_size = op->inputs.size();
            if (std::find(op->inputnames.begin(), op->inputnames.end(), "attn_mask") != op->inputnames.end())
            {
                in_size -= 1;
            }
            if (in_size == 3)
            {
                q_l = op->inputs[0]->shape[batch_first ? 1 : 0];
                k_s = op->inputs[1]->shape[batch_first ? 1 : 0];
                v_s = op->inputs[2]->shape[batch_first ? 1 : 0];
            }
            else if (in_size == 2)
            {
                q_l = op->inputs[0]->shape[batch_first ? 1 : 0];
                k_s = op->inputs[1]->shape[batch_first ? 1 : 0];
                v_s = k_s;
            }
            else
            {
                q_l = op->inputs[0]->shape[batch_first ? 1 : 0];
                k_s = q_l;
                v_s = q_l;
            }
            num_heads = op->params.at("num_heads").i;
            embed_dim = op->params.at("embed_dim").i;
            Kdim = op->params.at("kdim").i;
            vdim = op->params.at("vdim").i;
            linear1 = q_l * embed_dim * embed_dim + k_s * embed_dim * Kdim + v_s * embed_dim * vdim;
            attention = q_l * k_s * embed_dim + 2 * q_l * k_s * num_heads + q_l * v_s * embed_dim;
            linerar2 = q_l * embed_dim * embed_dim;
            m.flops += linear1 + attention + linerar2;
            weights = embed_dim * embed_dim + embed_dim * Kdim + embed_dim * vdim + num_heads * vdim * embed_dim;
            in = q_l * embed_dim + k_s * Kdim + v_s * vdim;
            attention_m = q_l * embed_dim + k_s * Kdim + 2 * q_l * k_s + v_s * vdim;
            out = q_l * embed_dim;
            m.memory_access += weights + in + attention_m + out;
        }
        else if (op->type == "nn.MaxPool1d" || op->type == "F.max_pool1d" && op->inputs[0]->shape.size() >= 1)
        {
            int num_o, in_size, out_size, in_l, out_l, mem = 1;
            num_o = op->params.at("return_indices").b ? 2 : 1;
            in_size = op->inputs[0]->shape.size();
            in_l = op->inputs[0]->shape[in_size - 1];
            out_size = op->outputs[0]->shape.size();
            out_l = op->outputs[0]->shape[out_size - 1];
            for (int index = 0; index < in_size - 1; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += mem * (in_l + out_l * num_o);
        }
        else if (op->type == "nn.MaxPool2d" || op->type == "F.max_pool2d" && op->inputs[0]->shape.size() >= 2)
        {
            int num_o, in_size, out_size, in_h, in_w, out_h, out_w, mem = 1;
            num_o = op->params.at("return_indices").b ? 2 : 1;
            in_size = op->inputs[0]->shape.size();
            in_h = op->inputs[0]->shape[in_size - 2];
            in_w = op->inputs[0]->shape[in_size - 1];
            out_size = op->outputs[0]->shape.size();
            out_h = op->outputs[0]->shape[out_size - 2];
            out_w = op->outputs[0]->shape[out_size - 1];
            for (int index = 0; index < in_size - 2; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += mem * (in_h * in_w + out_h * out_w * num_o);
        }
        else if (op->type == "nn.MaxPool3d" || op->type == "F.max_pool3d" && op->inputs[0]->shape.size() >= 3)
        {
            int num_o, in_size, out_size, in_d, in_h, in_w, out_d, out_h, out_w, mem = 1;
            num_o = op->params.at("return_indices").b ? 2 : 1;
            in_size = op->inputs[0]->shape.size();
            in_d = op->inputs[0]->shape[in_size - 3];
            in_h = op->inputs[0]->shape[in_size - 2];
            in_w = op->inputs[0]->shape[in_size - 1];
            out_size = op->outputs[0]->shape.size();
            out_d = op->outputs[0]->shape[out_size - 3];
            out_h = op->outputs[0]->shape[out_size - 2];
            out_w = op->outputs[0]->shape[out_size - 1];
            for (int index = 0; index < in_size - 3; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += mem * (in_d * in_h * in_w + out_d * out_h * out_w * num_o);
        }
        else if (op->type == "nn.AvgPool1d" || op->type == "F.avg_pool1d" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, out_size, in_l, out_l, k_l, kernel_add, kernel_avg, mem = 1;
            in_size = op->inputs[0]->shape.size();
            in_l = op->inputs[0]->shape[in_size - 1];
            out_size = op->outputs[0]->shape.size();
            out_l = op->outputs[0]->shape[out_size - 1];
            k_l = op->params.at("kernel_size").i;
            kernel_add = k_l - 1;
            kernel_avg = 1;
            for (int index = 0; index < in_size - 1; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += (kernel_add + kernel_avg) * out_l * mem;
            m.memory_access += mem * (in_l + out_l);
        }
        else if (op->type == "nn.AvgPool2d" || op->type == "F.avg_pool2d" && op->inputs[0]->shape.size() >= 2)
        {
            int in_size, out_size, in_h, in_w, out_h, out_w, k_h, k_w, kernel_add, kernel_avg, mem = 1;
            in_size = op->inputs[0]->shape.size();
            in_h = op->inputs[0]->shape[in_size - 2];
            in_w = op->inputs[0]->shape[in_size - 1];
            out_size = op->outputs[0]->shape.size();
            out_h = op->outputs[0]->shape[out_size - 2];
            out_w = op->outputs[0]->shape[out_size - 1];
            k_h = op->params.at("kernel_size").ai[0];
            k_w = op->params.at("kernel_size").ai[1];
            kernel_add = k_h * k_w - 1;
            kernel_avg = 1;
            for (int index = 0; index < in_size - 2; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += (kernel_add + kernel_avg) * (out_h * out_w) * mem;
            m.memory_access += mem * (in_h * in_w + out_h * out_w);
        }
        else if (op->type == "nn.AvgPool3d" || op->type == "F.avg_pool3d" && op->inputs[0]->shape.size() >= 3)
        {
            int in_size, out_size, in_d, in_h, in_w, out_d, out_h, out_w, k_d, k_h, k_w, kernel_add, kernel_avg, mem = 1;
            in_size = op->inputs[0]->shape.size();
            in_d = op->inputs[0]->shape[in_size - 3];
            in_h = op->inputs[0]->shape[in_size - 2];
            in_w = op->inputs[0]->shape[in_size - 1];
            out_size = op->outputs[0]->shape.size();
            out_d = op->outputs[0]->shape[out_size - 3];
            out_h = op->outputs[0]->shape[out_size - 2];
            out_w = op->outputs[0]->shape[out_size - 1];
            k_d = op->params.at("kernel_size").ai[0];
            k_h = op->params.at("kernel_size").ai[1];
            k_w = op->params.at("kernel_size").ai[2];
            kernel_add = k_d * k_h * k_w - 1;
            kernel_avg = 1;
            for (int index = 0; index < in_size - 3; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += (kernel_add + kernel_avg) * (out_d * out_h * out_w) * mem;
            m.memory_access += mem * (in_d * in_h * in_w + out_d * out_h * out_w);
        }
        else if (op->type == "nn.BatchNorm1d" && op->inputs[0]->shape.size() == 3)
        {
            int in_n, in_c, in_l;
            bool affine;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_l = op->inputs[0]->shape[2];
            affine = op->params.at("affine").b;
            if (affine)
            {
                m.flops += 7 * in_n * in_c * in_l;
                m.memory_access += 2 * in_n * in_c * in_l;
            }
            else
            {
                m.flops += 5 * in_n * in_c * in_l;
                m.memory_access += 2 * in_n * in_c * in_l;
            }
        }
        else if (op->type == "nn.BatchNorm2d" && op->inputs[0]->shape.size() == 4)
        {
            int in_n, in_c, in_h, in_w;
            bool affine;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_h = op->inputs[0]->shape[2];
            in_w = op->inputs[0]->shape[3];
            affine = op->params.at("affine").b;
            if (affine)
            {
                m.flops += 7 * in_n * in_c * in_h * in_w;
                m.memory_access += 2 * in_n * in_c * in_h * in_w;
            }
            else
            {
                m.flops += 5 * in_n * in_c * in_h * in_w;
                m.memory_access += 2 * in_n * in_c * in_h * in_w;
            }
        }
        else if (op->type == "nn.BatchNorm3d" && op->inputs[0]->shape.size() == 5)
        {
            int in_n, in_c, in_d, in_h, in_w;
            bool affine;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_d = op->inputs[0]->shape[2];
            in_h = op->inputs[0]->shape[3];
            in_w = op->inputs[0]->shape[4];
            affine = op->params.at("affine").b;
            if (affine)
            {
                m.flops += 7 * in_n * in_c * in_d * in_h * in_w;
                m.memory_access += 2 * in_n * in_c * in_d * in_h * in_w;
            }
            else
            {
                m.flops += 5 * in_n * in_c * in_d * in_h * in_w;
                m.memory_access += 2 * in_n * in_c * in_d * in_h * in_w;
            }
        }
        else if (op->type == "F.batch_norm" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 5 * mem;
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.AdaptiveAvgPool1d" || op->type == "F.adaptive_avg_pool1d" && op->inputs[0]->shape.size() == 3)
        {
            int in_n, in_c, in_l, out_l, k_l, kernel_add, kernel_avg;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_l = op->inputs[0]->shape[2];
            out_l = op->params.at("output_size").i;
            if (out_l == 0)
            {
                k_l = 1;
                out_l = in_l;
            }
            else
            {
                k_l = (in_l + out_l - 1) / out_l;
            }
            kernel_add = k_l - 1;
            kernel_avg = 1;
            m.flops += (kernel_add + kernel_avg) * out_l * in_c * in_n;
            m.memory_access += in_n * in_c * (in_l + out_l);
        }
        else if (op->type == "nn.AdaptiveAvgPool2d" || op->type == "F.adaptive_avg_pool2d" && op->inputs[0]->shape.size() == 4)
        {
            int in_n, in_c, in_h, in_w, out_h, out_w, k_h, k_w, kernel_add, kernel_avg;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_h = op->inputs[0]->shape[2];
            in_w = op->inputs[0]->shape[3];
            out_h = op->params.at("output_size").ai[0];
            out_w = op->params.at("output_size").ai[1];
            if (out_h == 0)
            {
                k_h = 1;
                out_h = in_h;
            }
            else
            {
                k_h = (in_h + out_h - 1) / out_h;
            }
            if (out_w == 0)
            {
                k_w = 1;
                out_w = in_w;
            }
            else
            {
                k_w = (in_w + out_w - 1) / out_w;
            }
            kernel_add = k_h * k_w - 1;
            kernel_avg = 1;
            m.flops += (kernel_add + kernel_avg) * out_h * out_w * in_c;
            m.memory_access += in_n * in_c * (in_h * in_w + out_h * out_w);
        }
        else if (op->type == "nn.AdaptiveAvgPool3d" || op->type == "F.adaptive_avg_pool3d" && op->inputs[0]->shape.size() == 5)
        {
            int in_n, in_c, in_d, in_h, in_w, out_d, out_h, out_w, k_d, k_h, k_w, kernel_add, kernel_avg;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_d = op->inputs[0]->shape[2];
            in_h = op->inputs[0]->shape[3];
            in_w = op->inputs[0]->shape[4];
            out_d = op->params.at("output_size").ai[0];
            out_h = op->params.at("output_size").ai[1];
            out_w = op->params.at("output_size").ai[2];
            if (out_d == 0)
            {
                k_d = 1;
                out_d = in_d;
            }
            else
            {
                k_d = (in_d + out_d - 1) / out_d;
            }
            if (out_h == 0)
            {
                k_h = 1;
                out_h = in_h;
            }
            else
            {
                k_h = (in_h + out_h - 1) / out_h;
            }
            if (out_w == 0)
            {
                k_w = 1;
                out_w = in_w;
            }
            else
            {
                k_w = (in_w + out_w - 1) / out_w;
            }
            kernel_add = k_d * k_h * k_w - 1;
            kernel_avg = 1;
            m.flops += (kernel_add + kernel_avg) * out_d * out_h * out_w * in_c * in_n;
            m.memory_access += in_n * in_c * (in_d * in_h * in_w + out_d * out_h * out_w);
        }
        else if (op->type == "nn.AdaptiveMaxPool1d" || op->type == "F.adaptive_max_pool1d" && op->inputs[0]->shape.size() == 3)
        {
            int num_o, in_n, in_c, in_l, out_l;
            num_o = op->params.at("return_indices").b ? 2 : 1;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_l = op->inputs[0]->shape[2];
            out_l = op->params.at("output_size").i;
            if (out_l == 0)
            {
                out_l = in_l;
            }
            m.memory_access += in_n * in_c * (in_l + out_l * num_o);
        }
        else if (op->type == "nn.AdaptiveMaxPool2d" || op->type == "F.adaptive_max_pool2d" && op->inputs[0]->shape.size() == 4)
        {
            int num_o, in_n, in_c, in_h, in_w, out_h, out_w;
            num_o = op->params.at("return_indices").b ? 2 : 1;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_h = op->inputs[0]->shape[2];
            in_w = op->inputs[0]->shape[3];
            out_h = op->params.at("output_size").ai[0];
            out_w = op->params.at("output_size").ai[1];
            if (out_h == 0)
            {
                out_h = in_h;
            }
            if (out_w == 0)
            {
                out_w = in_w;
            }
            m.memory_access += in_n * in_c * (in_h * in_w + out_h * out_w * num_o);
        }
        else if (op->type == "nn.AdaptiveMaxPool3d" || op->type == "F.adaptive_max_pool3d" && op->inputs[0]->shape.size() == 5)
        {
            int num_o, in_n, in_c, in_d, in_h, in_w, out_d, out_h, out_w;
            num_o = op->params.at("return_indices").b ? 2 : 1;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_d = op->inputs[0]->shape[2];
            in_h = op->inputs[0]->shape[3];
            in_w = op->inputs[0]->shape[4];
            out_d = op->params.at("output_size").ai[0];
            out_h = op->params.at("output_size").ai[1];
            out_w = op->params.at("output_size").ai[2];
            if (out_d == 0)
            {
                out_d = in_d;
            }
            if (out_h == 0)
            {
                out_h = in_h;
            }
            if (out_w == 0)
            {
                out_w = in_w;
            }
            m.memory_access += in_n * in_c * (in_d * in_h * in_w + out_d * out_h * out_w * num_o);
        }
        else if (op->type == "nn.CELU" || op->type == "F.celu" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.ELU" || op->type == "F.elu" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.Embedding" || op->type == "F.embedding" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.Fold" || op->type == "F.fold" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.GELU" || op->type == "F.gelu" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 12 * mem;
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.GLU" || op->type == "F.glu" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += std::round(2.5 * mem);
            m.memory_access += std::round(1.5 * mem);
        }
        else if (op->type == "nn.GroupNorm" && op->inputs[0]->shape.size() >= 2)
        {
            int num_g, in_n, in_c, in_size, mem = 1;
            num_g = op->params.at("num_groups").i;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_size = op->inputs[0]->shape.size();
            for (int index = 2; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 9 * in_n * in_c * mem + 2 * in_n * num_g;
            m.memory_access += 2 * in_n * in_c * mem + 2 * in_n * num_g;
        }
        else if (op->type == "F.group_norm" && op->inputs[0]->shape.size() >= 2)
        {
            int num_g, in_n, in_c, in_size, mem = 1;
            num_g = op->params.at("num_groups").i;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_size = op->inputs[0]->shape.size();
            for (int index = 2; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 9 * in_n * in_c * mem + 2 * in_n * num_g;
            m.memory_access += 2 * in_n * in_c * mem + 2 * in_n * num_g;
        }
        else if (op->type == "nn.GRU" && op->inputs[0]->shape.size() == 3)
        {
            int in_size, h_size, num_layers, batch, seq;
            bool batch_first, bidirectional;
            in_size = op->params.at("input_size").i;
            h_size = op->params.at("hidden_size").i;
            num_layers = op->params.at("num_layers").i;
            batch_first = op->params.at("batch_first").b;
            bidirectional = op->params.at("bidirectional").b;
            batch = op->inputs[0]->shape[batch_first ? 0 : 1];
            seq = op->inputs[0]->shape[batch_first ? 1 : 0];
            if (bidirectional)
            {
                m.flops += 2 * num_layers * batch * seq * h_size * (3 * in_size + 7);
            }
            else
            {
                m.flops += num_layers * batch * seq * h_size * (3 * in_size + 7);
            }
            m.memory_access += num_layers * batch * seq * in_size;
        }
        else if (op->type == "nn.Hardsigmoid" || op->type == "F.hardsigmoid" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 2 * mem;
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.Hardswish" || op->type == "F.hardswish" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 3 * mem;
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.Hardtanh" || op->type == "F.hardtanh" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.Identity" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.InstanceNorm1d" && op->inputs[0]->shape.size() == 3)
        {
            int in_n, in_c, in_l;
            bool affine;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_l = op->inputs[0]->shape[2];
            affine = op->params.at("affine").b;
            if (affine)
            {
                m.flops += 7 * in_n * in_c * in_l;
                m.memory_access += 2 * in_n * in_c * in_l;
            }
            else
            {
                m.flops += 5 * in_n * in_c * in_l;
                m.memory_access += 2 * in_n * in_c * in_l;
            }
        }
        else if (op->type == "nn.InstanceNorm2d" && op->inputs[0]->shape.size() == 4)
        {
            int in_n, in_c, in_h, in_w;
            bool affine;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_h = op->inputs[0]->shape[2];
            in_w = op->inputs[0]->shape[3];
            affine = op->params.at("affine").b;
            if (affine)
            {
                m.flops += 7 * in_n * in_c * in_h * in_w;
                m.memory_access += 2 * in_n * in_c * in_h * in_w;
            }
            else
            {
                m.flops += 5 * in_n * in_c * in_h * in_w;
                m.memory_access += 2 * in_n * in_c * in_h * in_w;
            }
        }
        else if (op->type == "nn.InstanceNorm3d" && op->inputs[0]->shape.size() == 5)
        {
            int in_n, in_c, in_d, in_h, in_w;
            bool affine;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_d = op->inputs[0]->shape[2];
            in_h = op->inputs[0]->shape[3];
            in_w = op->inputs[0]->shape[4];
            affine = op->params.at("affine").b;
            if (affine)
            {
                m.flops += 7 * in_n * in_c * in_d * in_h * in_w;
                m.memory_access += 2 * in_n * in_c * in_d * in_h * in_w;
            }
            else
            {
                m.flops += 5 * in_n * in_c * in_d * in_h * in_w;
                m.memory_access += 2 * in_n * in_c * in_d * in_h * in_w;
            }
        }
        else if (op->type == "F.instance_norm" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 5 * mem;
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.LeakyReLU" || op->type == "F.leaky_relu" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.LocalResponseNorm" && op->inputs[0]->shape.size() >= 2)
        {
            int in_size, mem = 1, size, in_n, in_c;
            size = op->params.at("size").i;
            in_size = op->inputs[0]->shape.size();
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            for (int index = 2; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += (size + 4) * in_n * in_c * mem;
            m.memory_access += (2 + size) * in_n * in_c * mem;
        }
        else if (op->type == "nn.LogSigmoid" || op->type == "F.logsigmoid" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 10 * mem;
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.LogSoftmax" || op->type == "F.log_softmax" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1, dim, in_n, in_c, in_h;
            dim = op->params.at("dim").i;
            in_size = op->inputs[0]->shape.size();
            if (dim == 0)
            {
                in_n = op->inputs[0]->shape[0];
                for (int index = 1; index < in_size; index++)
                {
                    mem *= op->inputs[0]->shape[index];
                }
                m.flops += (7 * in_n + 4) * mem;
                m.memory_access += 2 * in_n * mem;
            }
            else if (dim == 1)
            {
                in_n = op->inputs[0]->shape[0];
                in_c = op->inputs[0]->shape[1];
                for (int index = 2; index < in_size; index++)
                {
                    mem *= op->inputs[0]->shape[index];
                }
                m.flops += (7 * in_c + 4) * in_n * mem;
                m.memory_access += 2 * in_n * in_c * mem;
            }
            else if (dim == 2)
            {
                in_n = op->inputs[0]->shape[0];
                in_c = op->inputs[0]->shape[1];
                in_h = op->inputs[0]->shape[2];
                for (int index = 3; index < in_size; index++)
                {
                    mem *= op->inputs[0]->shape[index];
                }
                m.flops += (7 * in_h + 4) * in_n * in_c * mem;
                m.memory_access += 2 * in_n * in_c * in_h * mem;
            }
        }
        else if (op->type == "nn.LSTM" && op->inputs[0]->shape.size() == 3)
        {
            int hidden_size, num_layers, batch, seq, in_d;
            bool batch_first;
            hidden_size = op->params.at("hidden_size").i;
            num_layers = op->params.at("num_layers").i;
            batch_first = op->params.at("batch_first").b;
            batch = op->inputs[0]->shape[batch_first ? 0 : 1];
            seq = op->inputs[0]->shape[batch_first ? 1 : 0];
            in_d = op->inputs[0]->shape[2];
            m.flops += num_layers * batch * seq * hidden_size * (8 * (in_d + hidden_size) + 23);
            m.memory_access += num_layers * batch * (seq * in_d + 12 * seq * hidden_size);
        }
        else if (op->type == "nn.Mish" || op->type == "F.mish" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 5 * mem;
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.PixelShuffle" || op->type == "F.pixel_shuffle" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.PixelUnshuffle" || op->type == "F.pixel_unshuffle" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.ReflectionPad1d" && op->inputs[0]->shape.size() >= 1)
        {
            int pad_left, pad_right, in_size, in_w, mem = 1;
            pad_left = op->params.at("padding").ai[0];
            pad_right = op->params.at("padding").ai[1];
            in_size = op->inputs[0]->shape.size();
            in_w = op->inputs[0]->shape[in_size - 1];
            for (int index = 0; index < in_size - 1; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += mem * in_w + mem * (in_w + pad_left + pad_right);
        }
        else if (op->type == "nn.ReflectionPad2d" && op->inputs[0]->shape.size() >= 2)
        {
            int pad_left, pad_right, pad_top, pad_bottom, in_size, in_w, in_h, mem = 1;
            pad_left = op->params.at("padding").ai[0];
            pad_right = op->params.at("padding").ai[1];
            pad_top = op->params.at("padding").ai[2];
            pad_bottom = op->params.at("padding").ai[3];
            in_size = op->inputs[0]->shape.size();
            in_h = op->inputs[0]->shape[in_size - 2];
            in_w = op->inputs[0]->shape[in_size - 1];
            for (int index = 0; index < in_size - 2; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += mem * in_w * in_h + mem * (in_w + pad_left + pad_right) * (in_h + pad_top + pad_bottom);
        }
        else if (op->type == "nn.ReLU" || op->type == "F.relu" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.ReLU6" || op->type == "F.relu6" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.ReplicationPad1d" && op->inputs[0]->shape.size() == 3)
        {
            int pad_left, pad_right, in_size, in_l, mem = 1;
            pad_left = op->params.at("padding").ai[0];
            pad_right = op->params.at("padding").ai[1];
            in_size = op->inputs[0]->shape.size();
            in_l = op->inputs[0]->shape[in_size - 1];
            for (int index = 0; index < in_size - 1; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += mem * in_l + mem * (in_l + pad_left + pad_right);
        }
        else if (op->type == "nn.ReplicationPad2d" && op->inputs[0]->shape.size() == 4)
        {
            int pad_left, pad_right, pad_top, pad_bottom, in_size, in_w, in_h, mem = 1;
            pad_left = op->params.at("padding").ai[0];
            pad_right = op->params.at("padding").ai[1];
            pad_top = op->params.at("padding").ai[2];
            pad_bottom = op->params.at("padding").ai[3];
            in_size = op->inputs[0]->shape.size();
            in_h = op->inputs[0]->shape[in_size - 2];
            in_w = op->inputs[0]->shape[in_size - 1];
            for (int index = 0; index < in_size - 2; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += mem * in_w * in_h + mem * (in_w + pad_left + pad_right) * (in_h + pad_top + pad_bottom);
        }
        else if (op->type == "nn.ReplicationPad3d" && op->inputs[0]->shape.size() == 5)
        {
            int pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom, in_size, in_d, in_h, in_w, mem = 1;
            pad_left = op->params.at("padding").ai[0];
            pad_right = op->params.at("padding").ai[1];
            pad_top = op->params.at("padding").ai[2];
            pad_bottom = op->params.at("padding").ai[3];
            pad_front = op->params.at("padding").ai[4];
            pad_back = op->params.at("padding").ai[5];
            in_size = op->inputs[0]->shape.size();
            in_d = op->inputs[0]->shape[in_size - 3];
            in_h = op->inputs[0]->shape[in_size - 2];
            in_w = op->inputs[0]->shape[in_size - 1];
            for (int index = 0; index < in_size - 3; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += mem * in_d * in_h * in_w + mem * (in_d + pad_front + pad_back) * (in_h + pad_top + pad_bottom) * (in_w + pad_left + pad_right);
        }
        else if (op->type == "nn.RNN" && op->inputs[0]->shape.size() == 3)
        {
            int in_size, h_size, num_layers, batch, seq;
            bool batch_first, bidirectional;
            in_size = op->params.at("input_size").i;
            h_size = op->params.at("hidden_size").i;
            num_layers = op->params.at("num_layers").i;
            batch_first = op->params.at("batch_first").b;
            bidirectional = op->params.at("bidirectional").b;
            batch = op->inputs[0]->shape[batch_first ? 0 : 1];
            seq = op->inputs[0]->shape[batch_first ? 1 : 0];
            if (bidirectional)
            {
                m.flops += 2 * batch * seq * 2 * (in_size * h_size + (num_layers - 1) * (h_size * h_size));
            }
            else
            {
                m.flops += batch * seq * 2 * (in_size * h_size + (num_layers - 1) * (h_size * h_size));
            }
            m.memory_access += batch * seq * (in_size + num_layers * h_size + h_size);
        }
        else if (op->type == "nn.SELU" || op->type == "F.selu" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.Sigmoid" || op->type == "F.sigmoid" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 7 * mem;
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.SiLU" || op->type == "F.silu" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 8 * mem;
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.Softmax" || op->type == "F.softmax" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 7 * mem - 1;
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.Softmax2d" && op->inputs[0]->shape.size() == 4)
        {
            int in_n, in_c, in_h, in_w;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_h = op->inputs[0]->shape[2];
            in_w = op->inputs[0]->shape[3];
            m.flops += in_n * in_c * (7 * in_h * in_w - 1);
            m.memory_access += 2 * in_n * in_c * in_h * in_w;
        }
        else if (op->type == "nn.Tanh" || op->type == "F.tanh" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.flops += 9 * mem;
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.Unfold" || op->type == "F.unfold" && op->inputs[0]->shape.size() >= 1)
        {
            int in_size, mem = 1;
            in_size = op->inputs[0]->shape.size();
            for (int index = 0; index < in_size; index++)
            {
                mem *= op->inputs[0]->shape[index];
            }
            m.memory_access += 2 * mem;
        }
        else if (op->type == "nn.UpsamplingBilinear2d" || op->type == "F.upsample_bilinear" && op->inputs[0]->shape.size() == 4)
        {
            int in_n, in_c, in_h, in_w;
            bool has_size = false;
            in_n = op->inputs[0]->shape[0];
            in_c = op->inputs[0]->shape[1];
            in_h = op->inputs[0]->shape[2];
            in_w = op->inputs[0]->shape[3];
            if (op->params.find("size") != op->params.end())
            {
                std::string val = Parameter::encode_to_string(op->params.at("size"));
                if (val == "None")
                {
                    has_size = false;
                }
                else
                {
                    has_size = true;
                }
            }
            if (has_size)
            {
                int size_h = op->params.at("size").ai[0];
                int size_w = op->params.at("size").ai[1];
                m.flops += 4 * in_c * size_h * size_w;
                m.memory_access += 5 * in_c * size_h * size_w;
            }
            else
            {
                int scale_h = op->params.at("scale_factor").af[0];
                int scale_w = op->params.at("scale_factor").af[1];
                int out_h = in_h * scale_h;
                int out_w = in_w * scale_w;
                m.flops += 4 * in_c * out_h * out_w;
                m.memory_access += 5 * in_c * out_h * out_w;
            }
        }
    }

    return m;
}

} // namespace pnnx
