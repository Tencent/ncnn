#ifndef PYBIND11_NCNN_MODELBIN_H
#define PYBIND11_NCNN_MODELBIN_H

#include <modelbin.h>

template <class Base = ncnn::ModelBin>
class PyModelBin : public Base
{
public:
    using Base::Base; // Inherit constructors
    ncnn::Mat load(int w, int type) const override
    {
        PYBIND11_OVERLOAD_PURE(ncnn::Mat, Base, load, w, type);
    }
    //ncnn::Mat load(int w, int h, int type) const override {
    //	PYBIND11_OVERLOAD(ncnn::Mat, Base, load, w, h, type);
    //}
    //ncnn::Mat load(int w, int h, int c, int type) const override {
    //	PYBIND11_OVERLOAD(ncnn::Mat, Base, load, w, h, c, type);
    //}
};

template <class Other>
class PyModelBinOther : public PyModelBin<Other>
{
public:
    using PyModelBin<Other>::PyModelBin;
    ncnn::Mat load(int w, int type) const override
    {
        PYBIND11_OVERLOAD(ncnn::Mat, Other, load, w, type);
    }
};

#endif