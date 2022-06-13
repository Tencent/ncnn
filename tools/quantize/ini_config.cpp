#include "ini_config.h"
#include <sstream>
#include <cassert>

namespace ini {

template<typename T>
void Value::set(T val)
{
    text = std::to_string(f);
}

void Value::set(std::string str)
{
    text = '\"' + str + '\"';
}

template<typename T>
void Value::set(const std::vector<T>& data)
{
    text = "[ ";

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
}

template<typename T>
T Value::get()
{
    T result;
    std::stringstream ss;
    ss << text;
    ss >> result;
    return result;
}

template<typename T>
std::vector<T> Value::get()
{
    std::vector<T> result;

    std::string no_brace;
    {
        // remove brace
        auto start = text.find('[');
        auto end = text.find(']');
        no_brace = text.substr(start + 1, end);
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

            std::string val_str = no_brace.substr(start, end);
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

std::string Value::stringify()
{
    return text;
}

void Table::feed(std::string line)
{
    auto pos = line.find(':');
    assert(pos != std::string::npos);

    std::string key = line.substr(0, pos - 1);
    std::string value_str = line.substr(pos + 1);

    values[key] = std::make_shared<Value>(value_str);
}

void Table::feed(const std::vector<std::string>& lines)
{
    for (auto& line : lines)
    {
        feed(line);
    }
}

void Table::append(std::string key, float data)
{
    auto pVal = std::make_shared<Value>();
    pVal->set(data);
    values[key] = pVal;
}

void Table::append(std::string key, const std::vector<float>& data)
{
    auto pVal = std::make_shared<Value>();
    pVal->set(data);
    values[key] = pVal;
}

void Table::append(std::string key, std::string data)
{
    auto pVal = std::make_shared<Value>();
    pVal->set(data);
    values[key] = pVal;
}

std::shared_ptr<Value> Table::operator[](std::string key)
{
    return values[key];
}

std::string Table::stringify()
{
    std::string result;
    for (auto itra = values.begin(); itra != values.end(); ++itra)
    {
        result += itra->first;
        result += " = ";
        result += itra->second->stringify();
        result += '\n';
    }
    return result;
}

void Config::read(std::string path)
{
    std::ifstream fin;
    fin.open(path, std::ios::in);

    if (!fin.is_open())
    {
        fprintf(stderr, "open %s failed\n", path.c_str());
        return;
    }

    bool recoding = false;
    std::shared_ptr<Table> pTable = nullptr;

    std::string line;
    while (fin >> line)
    {
        if (nullptr == pTable)
        {
            auto start = line.find('[');
            auto end = line.find(']');
            assert(start != std::string::npos);
            assert(end != std::string::npos);

            std::string key = line.substr(start + 1, end);
            pTable = std::make_shared<Table>();
            tables[key] = pTable;
            continue;
        }

        if (line.length() <= 2)
        {
            pTable = nullptr;
            continue;
            ;
        }

        pTable->feed(line);
    }
}

std::vector<std::string> Config::list_all()
{
    std::vector<std::string> result;
    for (auto itra = tables.begin(); itra != tables.end(); ++itra)
    {
        result.push_back(itra->first);
    }
    return result;
}

std::shared_ptr<Table> Config::operator[](std::string key)
{
    return tables[key];
}

void Config::append(std::string key, std::shared_ptr<Table> table)
{
    tables[key] = table;
}

void Config::write(std::string path)
{
    // TODO
}

} // namespace ini
