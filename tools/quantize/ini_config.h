#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <memory>

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
namespace ini
{

class Value {
public:
    Value() {}

    void set(std::string str);

    template<typename T>
    void set(T data);
    
    template<typename T>
    void set(const std::vector<T>& data);

    template<typename T>
    T get();

    template<typename T>
    std::vector<T> get();

    std::string stringify();

private:
    std::string text;
};

class Table {
public:
    Table() {}

    void feed(std::string line);
    void feed(const std::vector<std::string>& lines);

    std::shared_ptr<Value> operator[](std::string key);

    void append(std::string key, float data);
    void append(std::string key, const std::vector<float>& data);
    void append(std::string key, std::string data);

    std::string stringify();

private:
    std::unordered_map<std::string, std::shared_ptr<Value> > values;
};

class Config {
public:
    Config() {}
    void read(std::string path);

    std::vector<std::string> list_all();
    std::shared_ptr<Table> operator[](std::string key);

    void append(std::string key, std::shared_ptr<Table> table);
    void write(std::string path);

private:
    std::unordered_map<std::string, std::shared_ptr<Table> > tables;
};

} // namespace ini
