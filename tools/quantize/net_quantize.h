#pragma once
// ncnn private header
#include "../modelwriter.h"

class NetQuantize : public ModelWriter
{
public:
    NetQuantize() {}

    std::map<std::string, ncnn::Mat> blob_int8scale_table;
    std::map<std::string, ncnn::Mat> weight_int8scale_table;

public:
    bool read_raw_format(const char* path);
    bool read_ini_format(const char* path);
    int quantize_convolution();
    int quantize_convolutiondepthwise();
    int quantize_innerproduct();

    int fuse_requantize();
};