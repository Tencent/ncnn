#ifndef PYBIND11_NCNN_MAT_H
#define PYBIND11_NCNN_MAT_H

#include <string>

#include <pybind11/pybind11.h>

#include <mat.h>

std::string get_mat_format(const ncnn::Mat &m)
{
    std::string format;
    if (m.elemsize == 4)
    { //float or int32, so what?
        format = pybind11::format_descriptor<float>::format();
    }
    if (m.elemsize == 2)
    {
        format = "e";
    }
    if (m.elemsize == 1)
    { //int8 or uint8, so what?
        format = pybind11::format_descriptor<int8_t>::format();
    }
    return format;
}

#endif