// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PYBIND11_NCNN_DATAREADER_H
#define PYBIND11_NCNN_DATAREADER_H

#include <datareader.h>

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
#if NCNN_STRING
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
#endif // NCNN_STRING
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

template<class Base = ncnn::DataReader>
class PyDataReader : public Base
{
public:
    using Base::Base; // Inherit constructors
#if NCNN_STRING
    int scan(const char* format, void* p) const override
    {
        PYBIND11_OVERRIDE(int, Base, scan, format, p);
    }
#endif // NCNN_STRING
    size_t read(void* buf, size_t size) const override
    {
        PYBIND11_OVERRIDE(size_t, Base, read, buf, size);
    }
};

template<class Other>
class PyDataReaderOther : public PyDataReader<Other>
{
public:
    using PyDataReader<Other>::PyDataReader;
#if NCNN_STRING
    int scan(const char* format, void* p) const override
    {
        PYBIND11_OVERRIDE(int, Other, scan, format, p);
    }
#endif // NCNN_STRING
    size_t read(void* buf, size_t size) const override
    {
        PYBIND11_OVERRIDE(size_t, Other, read, buf, size);
    }
};

#endif
