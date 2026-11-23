// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "datareader.h"

#include <ctype.h>
#if NCNN_STRING
#include <stdio.h>
#endif
#include <string.h>

namespace ncnn {

DataReader::DataReader()
{
}

DataReader::~DataReader()
{
}

#if NCNN_STRING
int DataReader::scan(const char* /*format*/, void* /*p*/) const
{
    return 0;
}
#endif // NCNN_STRING

size_t DataReader::read(void* /*buf*/, size_t /*size*/) const
{
    return 0;
}

size_t DataReader::reference(size_t /*size*/, const void** /*buf*/) const
{
    return 0;
}

#if NCNN_STDIO
class DataReaderFromStdioPrivate
{
public:
    DataReaderFromStdioPrivate(FILE* _fp)
        : fp(_fp)
    {
    }
    FILE* fp;
};

DataReaderFromStdio::DataReaderFromStdio(FILE* _fp)
    : DataReader(), d(new DataReaderFromStdioPrivate(_fp))
{
}

DataReaderFromStdio::~DataReaderFromStdio()
{
    delete d;
}

DataReaderFromStdio::DataReaderFromStdio(const DataReaderFromStdio&)
    : d(0)
{
}

DataReaderFromStdio& DataReaderFromStdio::operator=(const DataReaderFromStdio&)
{
    return *this;
}

#if NCNN_STRING
int DataReaderFromStdio::scan(const char* format, void* p) const
{
    return fscanf(d->fp, format, p);
}
#endif // NCNN_STRING

size_t DataReaderFromStdio::read(void* buf, size_t size) const
{
    return fread(buf, 1, size, d->fp);
}
#endif // NCNN_STDIO

class DataReaderFromMemoryPrivate
{
public:
    DataReaderFromMemoryPrivate(const unsigned char*& _mem)
        : mem(_mem), remaining(0), bounded(false)
    {
    }
    const unsigned char*& mem;
    size_t remaining;
    bool bounded;
#if NCNN_STRING
    std::vector<char> text;
#endif
};

DataReaderFromMemory::DataReaderFromMemory(const unsigned char*& _mem)
    : DataReader(), d(new DataReaderFromMemoryPrivate(_mem))
{
}

DataReaderFromMemory::DataReaderFromMemory(const unsigned char*& _mem, size_t mem_size)
    : DataReader(), d(new DataReaderFromMemoryPrivate(_mem))
{
    d->remaining = mem_size;
    d->bounded = true;
}

DataReaderFromMemory::~DataReaderFromMemory()
{
    delete d;
}

DataReaderFromMemory::DataReaderFromMemory(const DataReaderFromMemory&)
    : d(0)
{
}

DataReaderFromMemory& DataReaderFromMemory::operator=(const DataReaderFromMemory&)
{
    return *this;
}

#if NCNN_STRING
int DataReaderFromMemory::scan(const char* format, void* p) const
{
    if (d->bounded && d->remaining == 0)
        return 0;

    const char* mem = (const char*)d->mem;
    if (d->bounded)
    {
        // prepare bounded text once because the input may not be null-terminated
        if (d->text.empty())
        {
            d->text.resize(d->remaining + 1);
            memcpy(&d->text[0], mem, d->remaining);
            d->text[d->remaining] = 0;
        }
        mem = &d->text[d->text.size() - d->remaining - 1];
    }

    size_t fmtlen = strlen(format);

    const size_t nlen = fmtlen + 4;
    char* format_with_n = new char[nlen];
    snprintf(format_with_n, nlen, "%s%%n", format);

    int nconsumed = 0;
    int nscan = sscanf(mem, format_with_n, p, &nconsumed);
    d->mem += nconsumed;
    if (d->bounded)
        d->remaining -= nconsumed;

    delete[] format_with_n;

    return nconsumed > 0 ? nscan : 0;
}
#endif // NCNN_STRING

size_t DataReaderFromMemory::read(void* buf, size_t size) const
{
    if (d->bounded && size > d->remaining)
        size = d->remaining;

    if (size != 0)
    {
        memcpy(buf, d->mem, size);
        d->mem += size;
        if (d->bounded)
            d->remaining -= size;
    }
    return size;
}

size_t DataReaderFromMemory::reference(size_t size, const void** buf) const
{
    if (d->bounded && size > d->remaining)
        return 0;

    *buf = d->mem;
    if (size != 0)
    {
        d->mem += size;
        if (d->bounded)
            d->remaining -= size;
    }
    return size;
}

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 9
class DataReaderFromAndroidAssetPrivate
{
public:
    DataReaderFromAndroidAssetPrivate(AAsset* _asset)
        : asset(_asset), mem(0)
    {
    }
    AAsset* asset;
    mutable const unsigned char* mem;
};

DataReaderFromAndroidAsset::DataReaderFromAndroidAsset(AAsset* _asset)
    : DataReader(), d(new DataReaderFromAndroidAssetPrivate(_asset))
{
}

DataReaderFromAndroidAsset::~DataReaderFromAndroidAsset()
{
    delete d;
}

DataReaderFromAndroidAsset::DataReaderFromAndroidAsset(const DataReaderFromAndroidAsset&)
    : d(0)
{
}

DataReaderFromAndroidAsset& DataReaderFromAndroidAsset::operator=(const DataReaderFromAndroidAsset&)
{
    return *this;
}

#if NCNN_STRING
int DataReaderFromAndroidAsset::scan(const char* format, void* p) const
{
    if (!d->mem)
    {
        off_t pos = AAsset_seek(d->asset, 0, SEEK_CUR);
        d->mem = (const unsigned char*)AAsset_getBuffer(d->asset);
        d->mem += pos;
    }

    // asset internal buffer may not be NULL-terminated
    // create a NULL-terminated string for sscanf
    std::string line;
    {
        off64_t remain_length = AAsset_getRemainingLength64(d->asset);
        // include all leading whitespace and the following line in the buffer
        // sscanf decides whether the conversion consumes that whitespace
        size_t offset = 0;
        while (offset < (size_t)remain_length && isspace(d->mem[offset]))
            offset++;
        const char* newline_pos = (const char*)memchr(d->mem + offset, '\n', (size_t)remain_length - offset);

        size_t line_length = newline_pos ? newline_pos - (const char*)d->mem : (size_t)remain_length;
        line = std::string((const char*)d->mem, line_length);
    }

    int fmtlen = strlen(format);

    char* format_with_n = new char[fmtlen + 3];
    sprintf(format_with_n, "%s%%n", format);

    int nconsumed = 0;
    int nscan = sscanf(line.c_str(), format_with_n, p, &nconsumed);
    d->mem += nconsumed;

    delete[] format_with_n;

    if (nconsumed == 0)
        return 0;

    AAsset_seek(d->asset, nconsumed, SEEK_CUR);

    return nscan;
}
#endif // NCNN_STRING

size_t DataReaderFromAndroidAsset::read(void* buf, size_t size) const
{
    int nread = AAsset_read(d->asset, buf, size);
    if (nread < 0)
        return 0;

    if (d->mem)
    {
        d->mem += nread;
    }

    return nread;
}
#endif // __ANDROID_API__ >= 9
#endif // NCNN_PLATFORM_API

} // namespace ncnn
