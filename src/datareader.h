// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_DATAREADER_H
#define NCNN_DATAREADER_H

#include "platform.h"
#if NCNN_STDIO
#include <stdio.h>
#endif

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 9
#include <android/asset_manager.h>
#endif
#endif // NCNN_PLATFORM_API

namespace ncnn {

// data read wrapper
class NCNN_EXPORT DataReader
{
public:
    DataReader();
    virtual ~DataReader();

#if NCNN_STRING
    // parse plain param text
    // return 1 if scan success
    virtual int scan(const char* format, void* p) const;
#endif // NCNN_STRING

    // read binary param and model data
    // return bytes read
    virtual size_t read(void* buf, size_t size) const;

    // get model data reference
    // return bytes referenced
    virtual size_t reference(size_t size, const void** buf) const;
};

#if NCNN_STDIO
class DataReaderFromStdioPrivate;
class NCNN_EXPORT DataReaderFromStdio : public DataReader
{
public:
    explicit DataReaderFromStdio(FILE* fp);
    virtual ~DataReaderFromStdio();

#if NCNN_STRING
    virtual int scan(const char* format, void* p) const;
#endif // NCNN_STRING
    virtual size_t read(void* buf, size_t size) const;

private:
    DataReaderFromStdio(const DataReaderFromStdio&);
    DataReaderFromStdio& operator=(const DataReaderFromStdio&);

private:
    DataReaderFromStdioPrivate* const d;
};
#endif // NCNN_STDIO

class DataReaderFromMemoryPrivate;
class NCNN_EXPORT DataReaderFromMemory : public DataReader
{
public:
    explicit DataReaderFromMemory(const unsigned char*& mem);
    virtual ~DataReaderFromMemory();

#if NCNN_STRING
    virtual int scan(const char* format, void* p) const;
#endif // NCNN_STRING
    virtual size_t read(void* buf, size_t size) const;
    virtual size_t reference(size_t size, const void** buf) const;

private:
    DataReaderFromMemory(const DataReaderFromMemory&);
    DataReaderFromMemory& operator=(const DataReaderFromMemory&);

private:
    DataReaderFromMemoryPrivate* const d;
};

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 9
class DataReaderFromAndroidAssetPrivate;
class NCNN_EXPORT DataReaderFromAndroidAsset : public DataReader
{
public:
    explicit DataReaderFromAndroidAsset(AAsset* asset);
    virtual ~DataReaderFromAndroidAsset();

#if NCNN_STRING
    virtual int scan(const char* format, void* p) const;
#endif // NCNN_STRING
    virtual size_t read(void* buf, size_t size) const;

private:
    DataReaderFromAndroidAsset(const DataReaderFromAndroidAsset&);
    DataReaderFromAndroidAsset& operator=(const DataReaderFromAndroidAsset&);

private:
    DataReaderFromAndroidAssetPrivate* const d;
};
#endif // __ANDROID_API__ >= 9
#endif // NCNN_PLATFORM_API

} // namespace ncnn

#endif // NCNN_DATAREADER_H
