#pragma once
// ncnn private header
#include "../modelwriter.h"

class NetQuantize : public ModelWriter
{
public:
    NetQuantize();

    std::map<std::string, ncnn::Mat> blob_int8scale_table;
    std::map<std::string, ncnn::Mat> weight_int8scale_table;

public:
    void visit(const cpptoml::value<std::string>& v);
    void visit(const cpptoml::table& t);
    void read_toml(const std::string& toml_path);
    int quantize_convolution();
    int quantize_convolutiondepthwise();
    int quantize_innerproduct();
    
    int fuse_requantize();
};
