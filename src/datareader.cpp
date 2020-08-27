// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "datareader.h"

#include <string.h>

namespace ncnn {

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

#if NCNN_STDIO
DataReaderFromStdio::DataReaderFromStdio(FILE* _fp)
    : fp(_fp)
{
}

#if NCNN_STRING
int DataReaderFromStdio::scan(const char* format, void* p) const
{
    return fscanf(fp, format, p);
}
#endif // NCNN_STRING

size_t DataReaderFromStdio::read(void* buf, size_t size) const
{
    return fread(buf, 1, size, fp);
}
#endif // NCNN_STDIO

DataReaderFromMemory::DataReaderFromMemory(const unsigned char*& _mem)
    : mem(_mem)
{
}

#if NCNN_STRING
int DataReaderFromMemory::scan(const char* format, void* p) const
{
    size_t fmtlen = strlen(format);

    char* format_with_n = new char[fmtlen + 4];
    sprintf(format_with_n, "%s%%n", format);

    int nconsumed = 0;
    int nscan = sscanf((const char*)mem, format_with_n, p, &nconsumed);
    mem += nconsumed;

    delete[] format_with_n;

    return nconsumed > 0 ? nscan : 0;
}
#endif // NCNN_STRING

size_t DataReaderFromMemory::read(void* buf, size_t size) const
{
    memcpy(buf, mem, size);
    mem += size;
    return size;
}

#if __ANDROID_API__ >= 9
DataReaderFromAndroidAsset::DataReaderFromAndroidAsset(AAsset* _asset)
    : asset(_asset), mem(0)
{
}

#if NCNN_STRING
int DataReaderFromAndroidAsset::scan(const char* format, void* p) const
{
    if (!mem)
    {
        off_t pos = AAsset_seek(asset, 0, SEEK_CUR);
        mem = (const unsigned char*)AAsset_getBuffer(asset);
        mem += pos;
    }

    int fmtlen = strlen(format);

    char* format_with_n = new char[fmtlen + 3];
    sprintf(format_with_n, "%s%%n", format);

    int nconsumed = 0;
    int nscan = sscanf((const char*)mem, format_with_n, p, &nconsumed);
    mem += nconsumed;

    delete[] format_with_n;

    if (nconsumed == 0)
        return 0;

    AAsset_seek(asset, nconsumed, SEEK_CUR);

    return nscan;
}
#endif // NCNN_STRING

size_t DataReaderFromAndroidAsset::read(void* buf, size_t size) const
{
    int nread = AAsset_read(asset, buf, size);
    if (nread < 0)
        return 0;

    if (mem)
    {
        mem += nread;
    }

    return nread;
}
#endif // __ANDROID_API__ >= 9

} // namespace ncnn
