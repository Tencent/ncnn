/* Tencent is pleased to support the open source community by making ncnn available.
 *
 * Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

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
