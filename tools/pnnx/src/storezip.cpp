// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "storezip.h"

#include <stdio.h>
#include <stdint.h>
#include <map>
#include <string>
#include <vector>

namespace pnnx {

// https://stackoverflow.com/questions/1537964/visual-c-equivalent-of-gccs-attribute-packed
#ifdef _MSC_VER
#define PACK(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#else
#define PACK(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif

PACK(struct local_file_header {
    uint16_t version;
    uint16_t flag;
    uint16_t compression;
    uint16_t last_modify_time;
    uint16_t last_modify_date;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t file_name_length;
    uint16_t extra_field_length;
});

PACK(struct central_directory_file_header {
    uint16_t version_made;
    uint16_t version;
    uint16_t flag;
    uint16_t compression;
    uint16_t last_modify_time;
    uint16_t last_modify_date;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t file_name_length;
    uint16_t extra_field_length;
    uint16_t file_comment_length;
    uint16_t start_disk;
    uint16_t internal_file_attrs;
    uint32_t external_file_attrs;
    uint32_t lfh_offset;
});

PACK(struct end_of_central_directory_record {
    uint16_t disk_number;
    uint16_t start_disk;
    uint16_t cd_records;
    uint16_t total_cd_records;
    uint32_t cd_size;
    uint32_t cd_offset;
    uint16_t comment_length;
});

static uint32_t CRC32_TABLE[256];

static void CRC32_TABLE_INIT()
{
    for (int i = 0; i < 256; i++)
    {
        uint32_t c = i;
        for (int j = 0; j < 8; j++)
        {
            if (c & 1)
                c = (c >> 1) ^ 0xedb88320;
            else
                c >>= 1;
        }
        CRC32_TABLE[i] = c;
    }
}

static uint32_t CRC32(uint32_t x, unsigned char ch)
{
    return (x >> 8) ^ CRC32_TABLE[(x ^ ch) & 0xff];
}

static uint32_t CRC32_buffer(const unsigned char* data, int len)
{
    uint32_t x = 0xffffffff;

    for (int i = 0; i < len; i++)
        x = CRC32(x, data[i]);

    return x ^ 0xffffffff;
}

StoreZipReader::StoreZipReader()
{
    fp = 0;
}

StoreZipReader::~StoreZipReader()
{
    close();
}

int StoreZipReader::open(const std::string& path)
{
    close();

    fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    while (!feof(fp))
    {
        // peek signature
        uint32_t signature;
        int nread = fread((char*)&signature, sizeof(signature), 1, fp);
        if (nread != 1)
            break;

        if (signature == 0x04034b50)
        {
            local_file_header lfh;
            fread((char*)&lfh, sizeof(lfh), 1, fp);

            if (lfh.flag & 0x08)
            {
                fprintf(stderr, "zip file contains data descriptor, this is not supported yet\n");
                return -1;
            }

            if (lfh.compression != 0 || lfh.compressed_size != lfh.uncompressed_size)
            {
                fprintf(stderr, "not stored zip file %d %d\n", lfh.compressed_size, lfh.uncompressed_size);
                return -1;
            }

            // file name
            std::string name;
            name.resize(lfh.file_name_length);
            fread((char*)name.data(), name.size(), 1, fp);

            // skip extra field
            fseek(fp, lfh.extra_field_length, SEEK_CUR);

            StoreZipMeta fm;
            fm.offset = ftell(fp);
            fm.size = lfh.compressed_size;

            filemetas[name] = fm;

            //             fprintf(stderr, "%s = %d  %d\n", name.c_str(), fm.offset, fm.size);

            fseek(fp, lfh.compressed_size, SEEK_CUR);
        }
        else if (signature == 0x02014b50)
        {
            central_directory_file_header cdfh;
            fread((char*)&cdfh, sizeof(cdfh), 1, fp);

            // skip file name
            fseek(fp, cdfh.file_name_length, SEEK_CUR);

            // skip extra field
            fseek(fp, cdfh.extra_field_length, SEEK_CUR);

            // skip file comment
            fseek(fp, cdfh.file_comment_length, SEEK_CUR);
        }
        else if (signature == 0x06054b50)
        {
            end_of_central_directory_record eocdr;
            fread((char*)&eocdr, sizeof(eocdr), 1, fp);

            // skip comment
            fseek(fp, eocdr.comment_length, SEEK_CUR);
        }
        else
        {
            fprintf(stderr, "unsupported signature %x\n", signature);
            return -1;
        }
    }

    return 0;
}

size_t StoreZipReader::get_file_size(const std::string& name)
{
    if (filemetas.find(name) == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return 0;
    }

    return filemetas[name].size;
}

int StoreZipReader::read_file(const std::string& name, char* data)
{
    if (filemetas.find(name) == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return -1;
    }

    size_t offset = filemetas[name].offset;
    size_t size = filemetas[name].size;

    fseek(fp, offset, SEEK_SET);
    fread(data, size, 1, fp);

    return 0;
}

int StoreZipReader::close()
{
    if (!fp)
        return 0;

    fclose(fp);
    fp = 0;

    return 0;
}

StoreZipWriter::StoreZipWriter()
{
    fp = 0;

    CRC32_TABLE_INIT();
}

StoreZipWriter::~StoreZipWriter()
{
    close();
}

int StoreZipWriter::open(const std::string& path)
{
    close();

    fp = fopen(path.c_str(), "wb");
    if (!fp)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    return 0;
}

int StoreZipWriter::write_file(const std::string& name, const char* data, size_t size)
{
    int offset = ftell(fp);

    uint32_t signature = 0x04034b50;
    fwrite((char*)&signature, sizeof(signature), 1, fp);

    uint32_t crc32 = CRC32_buffer((const unsigned char*)data, size);

    local_file_header lfh;
    lfh.version = 0;
    lfh.flag = 0;
    lfh.compression = 0;
    lfh.last_modify_time = 0;
    lfh.last_modify_date = 0;
    lfh.crc32 = crc32;
    lfh.compressed_size = size;
    lfh.uncompressed_size = size;
    lfh.file_name_length = name.size();
    lfh.extra_field_length = 0;

    fwrite((char*)&lfh, sizeof(lfh), 1, fp);

    fwrite((char*)name.c_str(), name.size(), 1, fp);

    fwrite(data, size, 1, fp);

    StoreZipMeta szm;
    szm.name = name;
    szm.lfh_offset = offset;
    szm.crc32 = crc32;
    szm.size = size;

    filemetas.push_back(szm);

    return 0;
}

int StoreZipWriter::close()
{
    if (!fp)
        return 0;

    int offset = ftell(fp);

    for (const StoreZipMeta& szm : filemetas)
    {
        uint32_t signature = 0x02014b50;
        fwrite((char*)&signature, sizeof(signature), 1, fp);

        central_directory_file_header cdfh;
        cdfh.version_made = 0;
        cdfh.version = 0;
        cdfh.flag = 0;
        cdfh.compression = 0;
        cdfh.last_modify_time = 0;
        cdfh.last_modify_date = 0;
        cdfh.crc32 = szm.crc32;
        cdfh.compressed_size = szm.size;
        cdfh.uncompressed_size = szm.size;
        cdfh.file_name_length = szm.name.size();
        cdfh.extra_field_length = 0;
        cdfh.file_comment_length = 0;
        cdfh.start_disk = 0;
        cdfh.internal_file_attrs = 0;
        cdfh.external_file_attrs = 0;
        cdfh.lfh_offset = szm.lfh_offset;

        fwrite((char*)&cdfh, sizeof(cdfh), 1, fp);

        fwrite((char*)szm.name.c_str(), szm.name.size(), 1, fp);
    }

    int offset2 = ftell(fp);

    {
        uint32_t signature = 0x06054b50;
        fwrite((char*)&signature, sizeof(signature), 1, fp);

        end_of_central_directory_record eocdr;
        eocdr.disk_number = 0;
        eocdr.start_disk = 0;
        eocdr.cd_records = filemetas.size();
        eocdr.total_cd_records = filemetas.size();
        eocdr.cd_size = offset2 - offset;
        eocdr.cd_offset = offset;
        eocdr.comment_length = 0;

        fwrite((char*)&eocdr, sizeof(eocdr), 1, fp);
    }

    fclose(fp);
    fp = 0;

    return 0;
}

} // namespace pnnx

#if 0
int main()
{
    StoreZipReader sz;

    sz.open("test.zip");

    std::vector<float> data1;
    sz.read_file("pnnx2.py", data1);

    std::vector<float> data2;
    sz.read_file("pnnx2.param", data2);

    sz.close();


    StoreZipWriter szw;

    szw.open("szw.zip");

    szw.write_file("a.py", data1);
    szw.write_file("zzzz.param", data2);

    szw.close();


    return 0;
}
#endif
