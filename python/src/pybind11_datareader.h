#ifndef PYBIND11_NCNN_DATAREADER_H
#define PYBIND11_NCNN_DATAREADER_H

#include <datareader.h>

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char *format, void *p) const
    {
        return 0;
    }
    virtual size_t read(void *buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

template <class Base = ncnn::DataReader>
class PyDataReader : public Base
{
public:
    using Base::Base; // Inherit constructors
    int scan(const char *format, void *p) const override
    {
        PYBIND11_OVERLOAD(int, Base, scan, format, p);
    }
    size_t read(void *buf, size_t size) const override
    {
        PYBIND11_OVERLOAD(size_t, Base, read, buf, size);
    }
};

template <class Other>
class PyDataReaderOther : public PyDataReader<Other>
{
public:
    using PyDataReader<Other>::PyDataReader;
    int scan(const char *format, void *p) const override
    {
        PYBIND11_OVERLOAD(int, Other, scan, format, p);
    }
    size_t read(void *buf, size_t size) const override
    {
        PYBIND11_OVERLOAD(size_t, Other, read, buf, size);
    }
};

#endif