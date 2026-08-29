// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_pt2_parse.h"

#include "json.hpp"
#include "storezip.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace pnnx {

// ---------------------------------------------------------------------------
// 工具：从 zip 读取整条 entry 为 std::string
// ---------------------------------------------------------------------------
static std::string read_zip_entry(StoreZipReader& zr, const std::string& name)
{
    uint64_t sz = zr.get_file_size(name);
    std::string buf;
    buf.resize(sz);
    zr.read_file(name, &buf[0]);
    return buf;
}

static std::string find_entry_by_suffix(StoreZipReader& zr, const std::string& suffix)
{
    std::vector<std::string> names = zr.get_names();
    for (size_t i = 0; i < names.size(); i++)
    {
        if (names[i].size() >= suffix.size() && names[i].compare(names[i].size() - suffix.size(), suffix.size(), suffix) == 0)
            return names[i];
    }
    return std::string();
}

static std::vector<float> bytes_to_floats(const std::string& raw)
{
    std::vector<float> f;
    if (raw.size() % sizeof(float) != 0)
        return f;
    f.resize(raw.size() / sizeof(float));
    memcpy(f.data(), raw.data(), raw.size());
    return f;
}

static std::vector<int> json_array_to_ints(const JsonValue& arr)
{
    std::vector<int> shape;
    if (arr.isArray())
    {
        for (size_t i = 0; i < arr.size(); i++)
        {
            JsonValue e = arr[i];
            if (e.isInt())
                shape.push_back((int)e.asInt());
            else if (e.isObject() && e["as_int"].isInt())
                shape.push_back((int)e["as_int"].asInt());
        }
    }
    return shape;
}

// JSON 值 -> Pt2Param（按 JSON 类型分发）
static Pt2Param json_to_param(const JsonValue& v)
{
    Pt2Param p;
    if (v.isBool())
    {
        p.type = 1; p.b = v.asBool();
    }
    else if (v.isInt())
    {
        p.type = 2; p.i = v.asInt();
    }
    else if (v.isDouble())
    {
        p.type = 3; p.f = v.asDouble();
    }
    else if (v.isString())
    {
        p.type = 4; p.s = v.asString();
    }
    else if (v.isArray())
    {
        // 整数列表 or 浮点列表
        bool all_int = true;
        for (size_t i = 0; i < v.size(); i++)
            if (!v[i].isInt() && !v[i].isBool()) { all_int = false; break; }
        if (all_int && v.size() > 0)
        {
            p.type = 5;
            for (size_t i = 0; i < v.size(); i++)
                p.ii.push_back(v[i].isInt() ? v[i].asInt() : (long long)(v[i].asBool() ? 1 : 0));
        }
        else
        {
            p.type = 6;
            for (size_t i = 0; i < v.size(); i++)
                p.ff.push_back(v[i].isDouble() ? v[i].asDouble() : (v[i].isInt() ? (double)v[i].asInt() : 0.0));
        }
    }
    else
    {
        p.type = 0; // null
    }
    return p;
}

// ---------------------------------------------------------------------------
// 解析阶段：规范 pnnx IR 中间 zip -> Pt2Graph
// ---------------------------------------------------------------------------
int parse_pt2_zip(const std::string& pt2path, Pt2Graph& g)
{
    StoreZipReader zr;
    if (zr.open(pt2path) != 0)
    {
        fprintf(stderr, "load_pt2: open %s failed\n", pt2path.c_str());
        return -1;
    }

    std::string model_json_name = find_entry_by_suffix(zr, "models/model.json");
    if (model_json_name.empty())
    {
        fprintf(stderr, "load_pt2: no model.json found in %s\n", pt2path.c_str());
        zr.close();
        return -1;
    }

    JsonValue root = parse_json(read_zip_entry(zr, model_json_name));
    if (!root.isObject())
    {
        fprintf(stderr, "load_pt2: model.json is not an object\n");
        zr.close();
        return -1;
    }

    // ---- 用户输入 ----
    JsonValue inputs = root["inputs"];
    if (inputs.isArray())
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            Pt2IO io;
            io.name = inputs[i]["name"].asString();
            io.shape = json_array_to_ints(inputs[i]["shape"]);
            if (inputs[i]["dtype"].isInt())
                io.dtype = (int)inputs[i]["dtype"].asInt();
            g.inputs.push_back(io);
        }
    }

    // ---- 用户输出 ----
    JsonValue outputs = root["outputs"];
    if (outputs.isArray())
    {
        for (size_t i = 0; i < outputs.size(); i++)
        {
            Pt2IO io;
            io.name = outputs[i]["name"].asString();
            io.shape = json_array_to_ints(outputs[i]["shape"]);
            if (outputs[i]["dtype"].isInt())
                io.dtype = (int)outputs[i]["dtype"].asInt();
            g.outputs.push_back(io);
        }
    }

    // ---- 节点 ----
    JsonValue nodes = root["nodes"];
    if (nodes.isArray())
    {
        for (size_t i = 0; i < nodes.size(); i++)
        {
            JsonValue n = nodes[i];
            Pt2Node pn;
            pn.name = n["name"].asString();
            pn.pnnx_type = n["pnnx_type"].asString();

            JsonValue ins = n["inputs"];
            if (ins.isArray())
                for (size_t j = 0; j < ins.size(); j++)
                    pn.inputs.push_back(ins[j].asString());

            JsonValue outs = n["outputs"];
            if (outs.isArray())
                for (size_t j = 0; j < outs.size(); j++)
                    pn.outputs.push_back(outs[j].asString());

            JsonValue osh = n["output_shapes"];
            if (osh.isArray())
                for (size_t j = 0; j < osh.size(); j++)
                    pn.output_shapes.push_back(json_array_to_ints(osh[j]));

            JsonValue params = n["params"];
            if (params.isObject())
            {
                for (std::map<std::string, JsonValue>::const_iterator it = params.object_value.begin();
                     it != params.object_value.end(); ++it)
                {
                    pn.params[it->first] = json_to_param(it->second);
                }
            }

            JsonValue attrs = n["attrs"];
            if (attrs.isObject())
            {
                for (std::map<std::string, JsonValue>::const_iterator it = attrs.object_value.begin();
                     it != attrs.object_value.end(); ++it)
                {
                    JsonValue a = it->second;
                    Pt2Attr attr;
                    attr.shape = json_array_to_ints(a["shape"]);
                    if (a["dtype"].isInt())
                        attr.dtype = (int)a["dtype"].asInt();
                    std::string dp = a["data_path"].asString();
                    if (!dp.empty())
                    {
                        std::string full = find_entry_by_suffix(zr, "/" + dp);
                        if (full.empty())
                            full = dp;
                        attr.data = bytes_to_floats(read_zip_entry(zr, full));
                    }
                    pn.attrs[it->first] = attr;
                }
            }

            g.nodes.push_back(pn);
        }
    }

    zr.close();
    return 0;
}

} // namespace pnnx
