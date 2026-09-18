// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "storezip.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#ifndef _MSC_VER
#include <sys/types.h>
#endif
#include <algorithm>
#include <limits>
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

PACK(struct zip64_extended_extra_field {
    uint64_t uncompressed_size;
    uint64_t compressed_size;
    uint64_t lfh_offset;
    uint32_t disk_number;
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

PACK(struct zip64_end_of_central_directory_record {
    uint64_t size_of_eocd64_m12;
    uint16_t version_made_by;
    uint16_t version_min_required;
    uint32_t disk_number;
    uint32_t start_disk;
    uint64_t cd_records;
    uint64_t total_cd_records;
    uint64_t cd_size;
    uint64_t cd_offset;
});

PACK(struct zip64_end_of_central_directory_locator {
    uint32_t eocdr64_disk_number;
    uint64_t eocdr64_offset;
    uint32_t disk_count;
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

static uint32_t CRC32_buffer(const unsigned char* data, uint64_t len)
{
    uint32_t x = 0xffffffff;

    for (uint64_t i = 0; i < len; i++)
        x = CRC32(x, data[i]);

    return x ^ 0xffffffff;
}

static int storezip_fseek(FILE* fp, int64_t offset, int origin)
{
#ifdef _MSC_VER
    return _fseeki64(fp, offset, origin);
#else
    return fseeko(fp, (off_t)offset, origin);
#endif
}

static int64_t storezip_ftell(FILE* fp)
{
#ifdef _MSC_VER
    return _ftelli64(fp);
#else
    return ftello(fp);
#endif
}

StoreZipReader::StoreZipReader()
{
    CRC32_TABLE_INIT();
    fp = 0;
}

StoreZipReader::~StoreZipReader()
{
    close();
}

int StoreZipReader::open(const std::string& path, bool quiet)
{
    close();
    filemetas.clear();

    fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        if (!quiet) fprintf(stderr, "open failed\n");
        return -1;
    }

    // Locate the end of central directory. Reading metadata from the central
    // directory also supports archives whose local headers use data
    // descriptors (general-purpose flag bit 3), as produced by PT2Archive.
    if (storezip_fseek(fp, 0, SEEK_END) != 0)
        return -1;
    const int64_t file_size = storezip_ftell(fp);
    if (file_size < 22)
        return -1;

    const size_t search_size = (size_t)std::min<int64_t>(file_size, 65557);
    const int64_t search_begin = file_size - search_size;
    std::vector<unsigned char> tail(search_size);
    if (storezip_fseek(fp, search_begin, SEEK_SET) != 0 || fread(tail.data(), 1, tail.size(), fp) != tail.size())
        return -1;

    int64_t eocd_offset = -1;
    end_of_central_directory_record eocdr;
    for (size_t pos = search_size - 22 + 1; pos > 0; pos--)
    {
        const size_t offset = pos - 1;
        uint32_t signature = 0;
        memcpy(&signature, tail.data() + offset, 4);
        if (signature != 0x06054b50)
            continue;
        memcpy(&eocdr, tail.data() + offset + 4, sizeof(eocdr));
        if (offset + 4 + sizeof(eocdr) + eocdr.comment_length != search_size)
            continue;
        eocd_offset = search_begin + offset;
        break;
    }

    if (eocd_offset < 0)
    {
        if (!quiet) fprintf(stderr, "zip end of central directory not found\n");
        return -1;
    }

    uint64_t record_count = eocdr.total_cd_records;
    uint64_t central_directory_offset = eocdr.cd_offset;
    const bool zip64 = eocdr.disk_number == 0xffff || eocdr.start_disk == 0xffff || eocdr.cd_records == 0xffff || eocdr.total_cd_records == 0xffff || eocdr.cd_size == 0xffffffff || eocdr.cd_offset == 0xffffffff;
    if (!zip64 && (eocdr.disk_number != 0 || eocdr.start_disk != 0 || eocdr.cd_records != eocdr.total_cd_records))
        return -1;
    if (zip64)
    {
        const int64_t locator_offset = eocd_offset - 4 - (int64_t)sizeof(zip64_end_of_central_directory_locator);
        if (locator_offset < 0)
            return -1;
        if (storezip_fseek(fp, locator_offset, SEEK_SET) != 0)
            return -1;
        uint32_t signature = 0;
        zip64_end_of_central_directory_locator locator;
        if (fread((char*)&signature, sizeof(signature), 1, fp) != 1 || signature != 0x07064b50 || fread((char*)&locator, sizeof(locator), 1, fp) != 1)
            return -1;
        if (locator.eocdr64_disk_number != 0 || locator.disk_count != 1 || locator.eocdr64_offset > (uint64_t)file_size || storezip_fseek(fp, (int64_t)locator.eocdr64_offset, SEEK_SET) != 0)
            return -1;
        zip64_end_of_central_directory_record eocdr64;
        if (fread((char*)&signature, sizeof(signature), 1, fp) != 1 || signature != 0x06064b50 || fread((char*)&eocdr64, sizeof(eocdr64), 1, fp) != 1)
            return -1;
        if (eocdr64.disk_number != 0 || eocdr64.start_disk != 0 || eocdr64.cd_records != eocdr64.total_cd_records)
            return -1;
        record_count = eocdr64.total_cd_records;
        central_directory_offset = eocdr64.cd_offset;
    }

    if (central_directory_offset > (uint64_t)file_size || storezip_fseek(fp, (int64_t)central_directory_offset, SEEK_SET) != 0)
        return -1;
    for (uint64_t record_index = 0; record_index < record_count; record_index++)
    {
        uint32_t signature = 0;
        central_directory_file_header cdfh;
        if (fread((char*)&signature, sizeof(signature), 1, fp) != 1 || signature != 0x02014b50 || fread((char*)&cdfh, sizeof(cdfh), 1, fp) != 1)
            return -1;
        if (cdfh.flag & 1)
        {
            if (!quiet) fprintf(stderr, "encrypted zip entries are not supported\n");
            return -1;
        }
        std::string name(cdfh.file_name_length, '\0');
        if (!name.empty() && fread(&name[0], name.size(), 1, fp) != 1)
            return -1;
        std::vector<unsigned char> extra(cdfh.extra_field_length);
        if (!extra.empty() && fread(extra.data(), extra.size(), 1, fp) != 1)
            return -1;
        if (storezip_fseek(fp, cdfh.file_comment_length, SEEK_CUR) != 0)
            return -1;
        const int64_t next_cdfh_offset = storezip_ftell(fp);
        if (next_cdfh_offset < 0 || next_cdfh_offset > file_size)
            return -1;

        uint64_t compressed_size = cdfh.compressed_size;
        uint64_t uncompressed_size = cdfh.uncompressed_size;
        uint64_t lfh_offset = cdfh.lfh_offset;
        if (compressed_size == 0xffffffff || uncompressed_size == 0xffffffff || lfh_offset == 0xffffffff)
        {
            size_t extra_offset = 0;
            bool found_zip64 = false;
            while (extra_offset + 4 <= extra.size())
            {
                uint16_t extra_id = 0;
                uint16_t extra_size = 0;
                memcpy(&extra_id, extra.data() + extra_offset, 2);
                memcpy(&extra_size, extra.data() + extra_offset + 2, 2);
                extra_offset += 4;
                if (extra_offset + extra_size > extra.size())
                    return -1;
                if (extra_id == 0x0001)
                {
                    size_t p = extra_offset;
                    if (uncompressed_size == 0xffffffff)
                    {
                        if (p + 8 > extra_offset + extra_size) return -1;
                        memcpy(&uncompressed_size, extra.data() + p, 8);
                        p += 8;
                    }
                    if (compressed_size == 0xffffffff)
                    {
                        if (p + 8 > extra_offset + extra_size) return -1;
                        memcpy(&compressed_size, extra.data() + p, 8);
                        p += 8;
                    }
                    if (lfh_offset == 0xffffffff)
                    {
                        if (p + 8 > extra_offset + extra_size) return -1;
                        memcpy(&lfh_offset, extra.data() + p, 8);
                    }
                    found_zip64 = true;
                    break;
                }
                extra_offset += extra_size;
            }
            if (!found_zip64)
                return -1;
        }
        if (cdfh.compression == 0 && compressed_size != uncompressed_size)
            return -1;

        if (lfh_offset > (uint64_t)file_size || storezip_fseek(fp, (int64_t)lfh_offset, SEEK_SET) != 0)
            return -1;
        local_file_header lfh;
        if (fread((char*)&signature, sizeof(signature), 1, fp) != 1 || signature != 0x04034b50 || fread((char*)&lfh, sizeof(lfh), 1, fp) != 1)
            return -1;
        if ((lfh.flag & 1) || lfh.compression != cdfh.compression)
            return -1;

        StoreZipMeta fm;
        const int64_t local_header_end = storezip_ftell(fp);
        if (local_header_end < 0)
            return -1;
        std::string local_name(lfh.file_name_length, '\0');
        if (!local_name.empty() && fread(&local_name[0], local_name.size(), 1, fp) != 1)
            return -1;
        if (local_name != name)
            return -1;
        if ((lfh.flag & 8) == 0 && (lfh.crc32 != cdfh.crc32 || lfh.compressed_size != cdfh.compressed_size || lfh.uncompressed_size != cdfh.uncompressed_size))
            return -1;
        const int64_t data_offset = local_header_end + lfh.file_name_length + lfh.extra_field_length;
        if ((uint64_t)data_offset > (uint64_t)file_size || compressed_size > (uint64_t)file_size - (uint64_t)data_offset)
            return -1;
        fm.offset = data_offset;
        fm.size = uncompressed_size;
        fm.crc32 = cdfh.crc32;
        fm.compression = cdfh.compression;
        if (filemetas.find(name) != filemetas.end())
            return -1;
        filemetas[name] = fm;

        if (storezip_fseek(fp, next_cdfh_offset, SEEK_SET) != 0)
            return -1;
    }

    return 0;
}

std::vector<std::string> StoreZipReader::get_names() const
{
    std::vector<std::string> names;
    for (std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.begin(); it != filemetas.end(); ++it)
    {
        names.push_back(it->first);
    }

    return names;
}

uint64_t StoreZipReader::get_file_size(const std::string& name) const
{
    if (filemetas.find(name) == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return 0;
    }

    return filemetas.at(name).size;
}

bool StoreZipReader::is_file_stored(const std::string& name) const
{
    std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.find(name);
    return it != filemetas.end() && it->second.compression == 0;
}

int StoreZipReader::read_file(const std::string& name, char* data)
{
    if (filemetas.find(name) == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return -1;
    }

    const StoreZipMeta& meta = filemetas[name];
    if (meta.compression != 0)
    {
        fprintf(stderr, "compressed zip entry is not supported %s\n", name.c_str());
        return -1;
    }

    uint64_t offset = meta.offset;
    uint64_t size = meta.size;
    uint32_t crc32 = meta.crc32;

    if (size > std::numeric_limits<size_t>::max() || storezip_fseek(fp, (int64_t)offset, SEEK_SET) != 0 || fread(data, 1, (size_t)size, fp) != size)
        return -1;
    if (CRC32_buffer((const unsigned char*)data, size) != crc32)
    {
        fprintf(stderr, "zip crc mismatch for %s\n", name.c_str());
        return -1;
    }

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

int StoreZipWriter::write_file(const std::string& name, const char* data, uint64_t size)
{
    long offset = ftell(fp);

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
    lfh.compressed_size = 0xffffffff;
    lfh.uncompressed_size = 0xffffffff;
    lfh.file_name_length = name.size();

    // zip64 extra field
    zip64_extended_extra_field zip64_eef;
    zip64_eef.uncompressed_size = size;
    zip64_eef.compressed_size = size;
    zip64_eef.lfh_offset = 0;
    zip64_eef.disk_number = 0;

    uint16_t extra_id = 0x0001;
    uint16_t extra_size = sizeof(zip64_eef);

    lfh.extra_field_length = sizeof(extra_id) + sizeof(extra_size) + sizeof(zip64_eef);

    fwrite((char*)&lfh, sizeof(lfh), 1, fp);

    fwrite((char*)name.c_str(), name.size(), 1, fp);

    fwrite((char*)&extra_id, sizeof(extra_id), 1, fp);
    fwrite((char*)&extra_size, sizeof(extra_size), 1, fp);
    fwrite((char*)&zip64_eef, sizeof(zip64_eef), 1, fp);

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

    long offset = ftell(fp);

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
        cdfh.compressed_size = 0xffffffff;
        cdfh.uncompressed_size = 0xffffffff;
        cdfh.file_name_length = szm.name.size();
        cdfh.file_comment_length = 0;
        cdfh.start_disk = 0xffff;
        cdfh.internal_file_attrs = 0;
        cdfh.external_file_attrs = 0;
        cdfh.lfh_offset = 0xffffffff;

        // zip64 extra field
        zip64_extended_extra_field zip64_eef;
        zip64_eef.uncompressed_size = szm.size;
        zip64_eef.compressed_size = szm.size;
        zip64_eef.lfh_offset = szm.lfh_offset;
        zip64_eef.disk_number = 0;

        uint16_t extra_id = 0x0001;
        uint16_t extra_size = sizeof(zip64_eef);

        cdfh.extra_field_length = sizeof(extra_id) + sizeof(extra_size) + sizeof(zip64_eef);

        fwrite((char*)&cdfh, sizeof(cdfh), 1, fp);

        fwrite((char*)szm.name.c_str(), szm.name.size(), 1, fp);

        fwrite((char*)&extra_id, sizeof(extra_id), 1, fp);
        fwrite((char*)&extra_size, sizeof(extra_size), 1, fp);
        fwrite((char*)&zip64_eef, sizeof(zip64_eef), 1, fp);
    }

    long offset2 = ftell(fp);

    {
        uint32_t signature = 0x06064b50;
        fwrite((char*)&signature, sizeof(signature), 1, fp);

        zip64_end_of_central_directory_record eocdr64;
        eocdr64.size_of_eocd64_m12 = sizeof(eocdr64) - 8;
        eocdr64.version_made_by = 0;
        eocdr64.version_min_required = 0;
        eocdr64.disk_number = 0;
        eocdr64.start_disk = 0;
        eocdr64.cd_records = filemetas.size();
        eocdr64.total_cd_records = filemetas.size();
        eocdr64.cd_size = offset2 - offset;
        eocdr64.cd_offset = offset;

        fwrite((char*)&eocdr64, sizeof(eocdr64), 1, fp);
    }

    {
        uint32_t signature = 0x07064b50;
        fwrite((char*)&signature, sizeof(signature), 1, fp);

        zip64_end_of_central_directory_locator eocdl64;
        eocdl64.eocdr64_disk_number = 0;
        eocdl64.eocdr64_offset = offset2;
        eocdl64.disk_count = 1;

        fwrite((char*)&eocdl64, sizeof(eocdl64), 1, fp);
    }

    {
        uint32_t signature = 0x06054b50;
        fwrite((char*)&signature, sizeof(signature), 1, fp);

        end_of_central_directory_record eocdr;
        eocdr.disk_number = 0xffff;
        eocdr.start_disk = 0xffff;
        eocdr.cd_records = 0xffff;
        eocdr.total_cd_records = 0xffff;
        eocdr.cd_size = 0xffffffff;
        eocdr.cd_offset = 0xffffffff;
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
    using namespace pnnx;

    {
        uint64_t len = 1*1024*1024*1024;
        // uint64_t len = 1*1024*1024;
        char* data1g = new char[len];

        StoreZipWriter szw;

        szw.open("szw.zip");

        szw.write_file("a.py", data1g, len);
        szw.write_file("b.param", data1g, 44);
        szw.write_file("c.bin", data1g, len);
        szw.write_file("d.txt", data1g, len);
        szw.write_file("e.jpg", data1g, len);
        szw.write_file("f.png", data1g, len);

        szw.close();

        delete[] data1g;
    }

    {
        StoreZipReader sz;

        sz.open("szw.zip");

        std::vector<std::string> names = sz.get_names();

        for (size_t i = 0; i < names.size(); i++)
        {
            uint64_t size = sz.get_file_size(names[i]);

            fprintf(stderr, "%s  %lu\n", names[i].c_str(), size);
        }

        sz.close();
    }

    return 0;
}
#endif
