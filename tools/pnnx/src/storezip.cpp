// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include "storezip.h"

#include <stdio.h>
#include <stdint.h>
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
    struct Initializer
    {
        Initializer()
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
    };
    static const Initializer initializer;
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

StoreZipReader::StoreZipReader()
{
    fp = 0;
}

StoreZipReader::~StoreZipReader()
{
    close();
}

static uint16_t zip_u16(const unsigned char* p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t zip_u32(const unsigned char* p)
{
    return (uint32_t)zip_u16(p) | ((uint32_t)zip_u16(p + 2) << 16);
}

static uint64_t zip_u64(const unsigned char* p)
{
    return (uint64_t)zip_u32(p) | ((uint64_t)zip_u32(p + 4) << 32);
}

static bool zip_seek(FILE* fp, uint64_t offset)
{
    if (offset > (uint64_t)std::numeric_limits<int64_t>::max())
        return false;
#ifdef _WIN32
    return _fseeki64(fp, (int64_t)offset, SEEK_SET) == 0;
#else
    if (offset > (uint64_t)std::numeric_limits<off_t>::max())
        return false;
    return fseeko(fp, (off_t)offset, SEEK_SET) == 0;
#endif
}

static bool zip_read(FILE* fp, uint64_t offset, void* data, size_t size, uint64_t end)
{
    return offset <= end && size <= end - offset && zip_seek(fp, offset)
           && (size == 0 || fread(data, 1, size, fp) == size);
}

static int zip_error(std::string& error, const char* message)
{
    error = message;
    return -1;
}

static bool zip64_extra(const std::vector<unsigned char>& extra, uint64_t& size, uint64_t& compressed_size, uint64_t& offset, uint32_t& disk)
{
    bool found = false;
    size_t pos = 0;
    while (pos < extra.size())
    {
        if (extra.size() - pos < 4)
            return false;
        const uint16_t id = zip_u16(&extra[pos]);
        const uint16_t length = zip_u16(&extra[pos + 2]);
        pos += 4;
        if (length > extra.size() - pos)
            return false;
        if (id == 1)
        {
            if (found)
                return false;
            found = true;
            size_t p = pos;
            const size_t end = pos + length;
            uint64_t* fields[] = {&size, &compressed_size, &offset};
            for (size_t i = 0; i < 3; i++)
            {
                if (*fields[i] != 0xffffffff)
                    continue;
                if (end - p < 8)
                    return false;
                *fields[i] = zip_u64(&extra[p]);
                p += 8;
            }
            if (disk == 0xffff)
            {
                if (end - p < 4)
                    return false;
                disk = zip_u32(&extra[p]);
            }
        }
        pos += length;
    }
    return found || (size != 0xffffffff && compressed_size != 0xffffffff && offset != 0xffffffff && disk != 0xffff);
}

int StoreZipReader::read_directory(uint64_t size, std::string& error)
{
    if (size < 22)
        return zip_error(error, "missing end of central directory");
    const size_t tail_size = size < 65557 ? (size_t)size : 65557;
    std::vector<unsigned char> tail(tail_size);
    if (!zip_read(fp, size - tail_size, tail.data(), tail.size(), size))
        return zip_error(error, "cannot read end of central directory");
    int64_t pos = (int64_t)tail_size - 22;
    for (; pos >= 0; pos--)
    {
        if (zip_u32(&tail[(size_t)pos]) == 0x06054b50 && (size_t)pos + 22 + zip_u16(&tail[(size_t)pos + 20]) == tail_size)
            break;
    }
    if (pos < 0)
        return zip_error(error, "missing or truncated end of central directory");
    const unsigned char* eocd = &tail[(size_t)pos];
    const uint64_t eocd_offset = size - tail_size + (size_t)pos;
    uint64_t count = zip_u16(eocd + 10);
    uint64_t cd_size = zip_u32(eocd + 12);
    uint64_t cd_offset = zip_u32(eocd + 16);
    // Entry counts can be exactly 65535 without ZIP64.
    const bool needs_zip64 = zip_u16(eocd + 4) == 0xffff || zip_u16(eocd + 6) == 0xffff
                             || cd_size == 0xffffffff || cd_offset == 0xffffffff;
    uint64_t directory_limit = eocd_offset;
    unsigned char locator[20];
    const bool has_zip64 = eocd_offset >= 20 && zip_read(fp, eocd_offset - 20, locator, sizeof(locator), size) && zip_u32(locator) == 0x07064b50;
    if (needs_zip64 && !has_zip64)
        return zip_error(error, "missing ZIP64 locator");
    if (has_zip64)
    {
        if (zip_u32(locator + 4) != 0 || zip_u32(locator + 16) != 1)
            return zip_error(error, "multi-volume ZIP64 is unsupported");
        const uint64_t offset = zip_u64(locator + 8);
        unsigned char record[56];
        if (!zip_read(fp, offset, record, sizeof(record), eocd_offset - 20) || zip_u32(record) != 0x06064b50)
            return zip_error(error, "invalid ZIP64 end of central directory");
        const uint64_t record_size = zip_u64(record + 4);
        if (record_size < 44 || record_size != eocd_offset - 20 - offset - 12)
            return zip_error(error, "invalid ZIP64 record size");
        if (zip_u32(record + 16) != 0 || zip_u32(record + 20) != 0 || zip_u64(record + 24) != zip_u64(record + 32))
            return zip_error(error, "multi-volume ZIP64 is unsupported");
        count = zip_u64(record + 32);
        cd_size = zip_u64(record + 40);
        cd_offset = zip_u64(record + 48);
        directory_limit = offset;
    }
    else if (zip_u16(eocd + 4) != 0 || zip_u16(eocd + 6) != 0 || zip_u16(eocd + 8) != count)
        return zip_error(error, "multi-volume zip is unsupported");

    if (cd_offset > directory_limit || cd_size > directory_limit - cd_offset || count > cd_size / 46)
        return zip_error(error, "central directory outside file");
    const uint64_t cd_end = cd_offset + cd_size;
    uint64_t cursor = cd_offset;
    for (uint64_t i = 0; i < count; i++)
    {
        unsigned char header[46];
        if (!zip_read(fp, cursor, header, sizeof(header), cd_end) || zip_u32(header) != 0x02014b50)
            return zip_error(error, "invalid central directory entry");
        cursor += sizeof(header);
        const uint16_t name_size = zip_u16(header + 28);
        const uint16_t extra_size = zip_u16(header + 30);
        const uint16_t comment_size = zip_u16(header + 32);
        const uint64_t variable_size = (uint64_t)name_size + extra_size + comment_size;
        if (variable_size > cd_end - cursor || name_size == 0)
            return zip_error(error, "invalid central directory entry lengths");
        std::string name(name_size, '\0');
        std::vector<unsigned char> extra(extra_size);
        if (!zip_read(fp, cursor, &name[0], name_size, cd_end) || !zip_read(fp, cursor + name_size, extra.data(), extra.size(), cd_end))
            return zip_error(error, "truncated central directory entry");
        cursor += variable_size;
        if (name.find('\0') != std::string::npos || filemetas.find(name) != filemetas.end())
            return zip_error(error, "invalid or duplicate entry name");

        uint64_t compressed_size = zip_u32(header + 20);
        uint64_t entry_size = zip_u32(header + 24);
        uint64_t local_offset = zip_u32(header + 42);
        uint32_t disk = zip_u16(header + 34);
        if (!zip64_extra(extra, entry_size, compressed_size, local_offset, disk))
            return zip_error(error, "invalid ZIP64 entry extra field");
        if (disk != 0)
            return zip_error(error, "multi-volume entry is unsupported");

        StoreZipMeta meta;
        meta.size = entry_size;
        meta.flag = zip_u16(header + 8);
        meta.compression = zip_u16(header + 10);
        meta.crc32 = zip_u32(header + 16);
        if (meta.compression == 0 && compressed_size != entry_size)
            return zip_error(error, "inconsistent stored entry sizes");

        unsigned char local[30];
        if (!zip_read(fp, local_offset, local, sizeof(local), cd_offset) || zip_u32(local) != 0x04034b50)
            return zip_error(error, "invalid local file header");
        const uint16_t local_name_size = zip_u16(local + 26);
        const uint16_t local_extra_size = zip_u16(local + 28);
        const uint64_t local_start = local_offset + sizeof(local);
        if ((uint64_t)local_name_size + local_extra_size > cd_offset - local_start
            || local_name_size != name_size || zip_u16(local + 6) != meta.flag || zip_u16(local + 8) != meta.compression)
            return zip_error(error, "inconsistent local file header");
        std::string local_name(local_name_size, '\0');
        if (!zip_read(fp, local_start, &local_name[0], local_name_size, cd_offset) || local_name != name)
            return zip_error(error, "inconsistent local entry name");
        // Sizes and CRC come from the central directory, including entries
        // whose local header leaves these fields for a data descriptor.
        meta.offset = local_start + local_name_size + local_extra_size;
        if (compressed_size > cd_offset - meta.offset)
            return zip_error(error, "entry data outside file");
        filemetas[name] = meta;
    }
    if (cursor != cd_end)
        return zip_error(error, "central directory size or entry count mismatch");
    return 0;
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

    std::string error;
#ifdef _WIN32
    const int seek_ret = _fseeki64(fp, 0, SEEK_END);
    const int64_t size = seek_ret == 0 ? _ftelli64(fp) : -1;
#else
    const int seek_ret = fseeko(fp, 0, SEEK_END);
    const int64_t size = seek_ret == 0 ? ftello(fp) : -1;
#endif
    if (size < 0 || read_directory((uint64_t)size, error) != 0)
    {
        fprintf(stderr, "invalid zip %s: %s\n", path.c_str(), size < 0 ? "cannot determine file size" : error.c_str());
        close();
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

bool StoreZipReader::has_file(const std::string& name) const
{
    return filemetas.find(name) != filemetas.end();
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

int StoreZipReader::read_file(const std::string& name, char* data)
{
    const std::map<std::string, StoreZipMeta>::const_iterator it = filemetas.find(name);
    if (!fp || it == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return -1;
    }

    const StoreZipMeta meta = it->second;
    if (meta.compression != 0 || (meta.flag & ~(uint16_t)0x808) != 0)
    {
        fprintf(stderr, "unsupported zip entry %s: compression %u flags 0x%x\n", name.c_str(), meta.compression, meta.flag);
        return -1;
    }
    if (meta.size > std::numeric_limits<size_t>::max() || (meta.size != 0 && !data) || !zip_seek(fp, meta.offset))
    {
        fprintf(stderr, "cannot read zip entry %s\n", name.c_str());
        return -1;
    }

    if (meta.size != 0 && fread(data, 1, (size_t)meta.size, fp) != meta.size)
    {
        fprintf(stderr, "short read in zip entry %s\n", name.c_str());
        return -1;
    }

    CRC32_TABLE_INIT();
    if (CRC32_buffer((const unsigned char*)data, meta.size) != meta.crc32)
    {
        fprintf(stderr, "CRC mismatch in zip entry %s\n", name.c_str());
        return -1;
    }

    return 0;
}

int StoreZipReader::close()
{
    filemetas.clear();
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
