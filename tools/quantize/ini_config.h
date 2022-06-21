// tpoisonooo is pleased to support the open source community by making ncnn available.
//
// author:tpoisonooo (https://github.com/tpoisonooo/) .
//
// Copyright (C) 2022 tpoisonooo. All rights reserved.
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

#pragma once
#include <cassert>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <map>
#include <vector>
#include <tuple>

// ini format table reader and writer
// file example:
//
// [Conv_0]
// type = "Conv"
// input_scale = 127.0
// weight = [ 1117.265625, 8819.232421875 ]
//
// [LayerNorm_66]
// type = "LayerNorm"
// zero_point = -24

namespace ini {

template<typename T>
std::string value_set(T data)
{
    return std::to_string(data);
}

template<>
std::string value_set<std::string>(std::string data);

template<>
std::string value_set<const char*>(const char* data);

template<typename T>
std::string value_set(const std::vector<T>& data)
{
    std::string text = "[ ";
    size_t len = data.size();
    if (len > 0)
    {
        size_t i = 0;
        for (; i < len - 1; ++i)
        {
            text += std::to_string(data[i]);
            text += ", ";
        }
        text += std::to_string(data[i]);
        text += " ";
    }
    text += "]";
    return text;
}

template<typename T>
T value_get(std::string text)
{
    T result;
    std::stringstream ss;
    ss << text;
    ss >> result;
    return result;
}

template<>
std::string value_get<std::string>(std::string text);

/**
 * @brief parse `[1, 2.2]` format to value list
 *
 * @tparam T
 * @param text
 * @return std::vector<T>
 */
template<typename T>
std::vector<T> value_get_list(std::string text)
{
    std::vector<T> result;
    std::string no_brace;
    {
        // remove brace
        auto start = text.find('[');
        auto end = text.find(']');
        no_brace = text.substr(start + 1, end - start - 1);
    }

    {
        // split with the separator ','
        std::stringstream ss;
        size_t end = 0, start = 0;
        while (true)
        {
            end = no_brace.find(',', start);
            if (end == std::string::npos)
            {
                break;
            }

            std::string val_str = no_brace.substr(start, end - start);
            start = end + 1;

            T val;
            ss << val_str;
            ss >> val;
            ss.clear();
            result.emplace_back(val);
        }

        // parse the last one
        std::string val_str = no_brace.substr(start);
        T val;
        ss << val_str;
        ss >> val;
        result.emplace_back(val);
    }
    return result;
}

/**
 * @brief contains multiple `key=value` lines
 *
 */
class Table
{
public:
    Table()
    {
    }

    void feed(std::string line)
    {
        auto pos = line.find('=');
        assert(pos != std::string::npos);

        std::string key = line.substr(0, pos - 1);
        std::string value_str = line.substr(pos + 2);

        values[key] = value_str;
    }

    void feed(const std::vector<std::string>& lines)
    {
        for (auto& line : lines)
        {
            feed(line);
        }
    }

    std::string operator[](std::string key)
    {
        return values[key];
    }

    template<typename T>
    T get(std::string key)
    {
        std::string text = values.at(key);
        return value_get<T>(text);
    }

    template<typename T>
    std::vector<T> get_list(std::string key)
    {
        std::string text = values[key];
        return value_get_list<T>(text);
    }

    template<typename T>
    void append(std::string key, T data)
    {
        values[key] = value_set(data);
    }

    template<typename T>
    void append(std::string key, const std::vector<T>& data)
    {
        values[key] = value_set(data);
    }

    std::string stringify()
    {
        std::string result;
        for (auto itra = values.begin(); itra != values.end(); ++itra)
        {
            result += itra->first;
            result += " = ";
            result += itra->second;
            result += '\n';
        }
        return result;
    }

private:
    std::map<std::string, std::string> values;
};

/**
 * @brief `Config` consist of multiple key-table
 *
 */
class Config
{
public:
    Config()
    {
    }

    void read(std::string path)
    {
        std::ifstream fin;
        fin.open(path, std::ios::in);

        if (!fin.is_open())
        {
            fprintf(stderr, "open %s failed\n", path.c_str());
            return;
        }

        std::shared_ptr<Table> pTable = nullptr;
        constexpr int BUF_LEN = 1024 * 1024;
        char buf[BUF_LEN] = {0};
        std::string line;
        while (!fin.eof())
        {
            fin.getline(buf, BUF_LEN);
            line = std::string(buf);

            if (line.length() <= 2)
            {
                pTable = nullptr;
                continue;
            }

            if (nullptr == pTable)
            {
                auto start = line.find('[');
                auto end = line.find(']');
                assert(start != std::string::npos);
                assert(end != std::string::npos);

                std::string key = line.substr(start + 1, end - start - 1);

                pTable = std::make_shared<Table>();
                append(key, pTable);
                continue;
            }

            pTable->feed(line);
        }

        fin.close();
    }

    std::vector<std::string> keys()
    {
        std::vector<std::string> result;
        for (auto& pair : tables)
        {
            result.push_back(std::get<0>(pair));
        }
        return result;
    }

    size_t size()
    {
        return tables.size();
    }

    std::tuple<std::string, std::shared_ptr<Table> > operator[](size_t i)
    {
        return tables[i];
    }

    void append(const std::string& key, std::shared_ptr<Table> table)
    {
        tables.emplace_back(std::make_pair(key, table));
    }

    void write(const std::string& path)
    {
        std::ofstream fout;
        fout.open(path, std::ios::out);
        if (!fout.is_open())
        {
            fprintf(stderr, "open %s failed\n", path.c_str());
        }

        for (auto& pair : tables)
        {
            std::string name = std::get<0>(pair);
            std::shared_ptr<Table> ptable = std::get<1>(pair);
            fout << "[" << name << "]\n";
            fout << ptable->stringify();
            fout << "\n";
        }
        fout.flush();
        fout.close();
    }

private:
    std::vector<std::tuple<std::string, std::shared_ptr<Table> > > tables;
};

} // namespace ini
