// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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

StoreZipReader::StoreZipReader()
{
    fp = 0;
}

StoreZipReader::~StoreZipReader()
{
    close();
}

static int file_seek(FILE* fp, int64_t offset, int origin)
{
#if _WIN32
    return _fseeki64(fp, offset, origin);
#else
    return fseeko(fp, offset, origin);
#endif
}

static int64_t file_tell(FILE* fp)
{
#if _WIN32
    return _ftelli64(fp);
#else
    return ftello(fp);
#endif
}

static bool read_zip64_value(const std::vector<unsigned char>& extra, size_t& offset, uint64_t& value)
{
    if (extra.size() - offset < sizeof(value))
        return false;

    memcpy(&value, extra.data() + offset, sizeof(value));
    offset += sizeof(value);
    return true;
}

int StoreZipReader::open(const std::string& path)
{
    close();
    filemetas.clear();

    fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    if (file_seek(fp, 0, SEEK_END) != 0)
    {
        fprintf(stderr, "seek failed\n");
        return -1;
    }

    const int64_t archive_size_i64 = file_tell(fp);
    if (archive_size_i64 < 22)
    {
        fprintf(stderr, "truncated zip file\n");
        return -1;
    }
    const uint64_t archive_size = (uint64_t)archive_size_i64;

    const uint64_t tail_size = std::min<uint64_t>(archive_size, 65557);
    std::vector<unsigned char> tail((size_t)tail_size);
    if (file_seek(fp, archive_size_i64 - (int64_t)tail_size, SEEK_SET) != 0 || fread(tail.data(), tail.size(), 1, fp) != 1)
    {
        fprintf(stderr, "read zip tail failed\n");
        return -1;
    }

    size_t eocd_offset = tail.size() - 4;
    bool found_eocd = false;
    while (true)
    {
        uint32_t signature = 0;
        memcpy(&signature, tail.data() + eocd_offset, sizeof(signature));
        if (signature == 0x06054b50 && eocd_offset + 4 + sizeof(end_of_central_directory_record) <= tail.size())
        {
            end_of_central_directory_record candidate;
            memcpy(&candidate, tail.data() + eocd_offset + 4, sizeof(candidate));
            if (eocd_offset + 4 + sizeof(candidate) + candidate.comment_length == tail.size())
            {
                found_eocd = true;
                break;
            }
        }
        if (eocd_offset == 0)
            break;
        eocd_offset--;
    }
    if (!found_eocd)
    {
        fprintf(stderr, "zip file has no end of central directory\n");
        return -1;
    }

    end_of_central_directory_record eocdr;
    memcpy(&eocdr, tail.data() + eocd_offset + 4, sizeof(eocdr));
    if (eocd_offset + 4 + sizeof(eocdr) + eocdr.comment_length != tail.size())
    {
        fprintf(stderr, "invalid zip end of central directory\n");
        return -1;
    }

    uint64_t central_directory_offset = eocdr.cd_offset;
    uint64_t record_count = eocdr.total_cd_records;
    if (eocdr.cd_offset == 0xffffffff || eocdr.total_cd_records == 0xffff)
    {
        const uint64_t absolute_eocd_offset = archive_size - tail_size + eocd_offset;
        if (absolute_eocd_offset < 4 + sizeof(zip64_end_of_central_directory_locator))
        {
            fprintf(stderr, "zip64 end of central directory locator is missing\n");
            return -1;
        }

        if (file_seek(fp, (int64_t)(absolute_eocd_offset - 4 - sizeof(zip64_end_of_central_directory_locator)), SEEK_SET) != 0)
            return -1;
        uint32_t locator_signature = 0;
        zip64_end_of_central_directory_locator locator;
        if (fread(&locator_signature, sizeof(locator_signature), 1, fp) != 1 || locator_signature != 0x07064b50 || fread(&locator, sizeof(locator), 1, fp) != 1)
        {
            fprintf(stderr, "invalid zip64 end of central directory locator\n");
            return -1;
        }

        if (locator.eocdr64_offset > archive_size - 4 - sizeof(zip64_end_of_central_directory_record) || file_seek(fp, (int64_t)locator.eocdr64_offset, SEEK_SET) != 0)
            return -1;
        uint32_t zip64_signature = 0;
        zip64_end_of_central_directory_record zip64_eocdr;
        if (fread(&zip64_signature, sizeof(zip64_signature), 1, fp) != 1 || zip64_signature != 0x06064b50 || fread(&zip64_eocdr, sizeof(zip64_eocdr), 1, fp) != 1)
        {
            fprintf(stderr, "invalid zip64 end of central directory\n");
            return -1;
        }
        central_directory_offset = zip64_eocdr.cd_offset;
        record_count = zip64_eocdr.total_cd_records;
    }

    if (central_directory_offset > archive_size || file_seek(fp, (int64_t)central_directory_offset, SEEK_SET) != 0)
    {
        fprintf(stderr, "invalid zip central directory offset\n");
        return -1;
    }

    for (uint64_t record_index = 0; record_index < record_count; record_index++)
    {
        uint32_t signature = 0;
        central_directory_file_header cdfh;
        if (fread(&signature, sizeof(signature), 1, fp) != 1 || signature != 0x02014b50 || fread(&cdfh, sizeof(cdfh), 1, fp) != 1)
        {
            fprintf(stderr, "invalid zip central directory entry\n");
            return -1;
        }

        std::string name(cdfh.file_name_length, '\0');
        std::vector<unsigned char> extra(cdfh.extra_field_length);
        if ((!name.empty() && fread(&name[0], name.size(), 1, fp) != 1) || (!extra.empty() && fread(extra.data(), extra.size(), 1, fp) != 1) || file_seek(fp, cdfh.file_comment_length, SEEK_CUR) != 0)
        {
            fprintf(stderr, "truncated zip central directory entry\n");
            return -1;
        }
        const int64_t next_record = file_tell(fp);

        uint64_t compressed_size = cdfh.compressed_size;
        uint64_t uncompressed_size = cdfh.uncompressed_size;
        uint64_t local_header_offset = cdfh.lfh_offset;
        if (compressed_size == 0xffffffff || uncompressed_size == 0xffffffff || local_header_offset == 0xffffffff)
        {
            bool found_zip64 = false;
            for (size_t extra_offset = 0; extra.size() - extra_offset >= 4;)
            {
                uint16_t extra_id = 0;
                uint16_t extra_size = 0;
                memcpy(&extra_id, extra.data() + extra_offset, sizeof(extra_id));
                memcpy(&extra_size, extra.data() + extra_offset + 2, sizeof(extra_size));
                extra_offset += 4;
                if (extra_size > extra.size() - extra_offset)
                    return -1;
                if (extra_id == 0x0001)
                {
                    const size_t extra_end = extra_offset + extra_size;
                    if ((uncompressed_size == 0xffffffff && !read_zip64_value(extra, extra_offset, uncompressed_size)) || (compressed_size == 0xffffffff && !read_zip64_value(extra, extra_offset, compressed_size)) || (local_header_offset == 0xffffffff && !read_zip64_value(extra, extra_offset, local_header_offset)) || extra_offset > extra_end)
                    {
                        fprintf(stderr, "invalid zip64 extra field\n");
                        return -1;
                    }
                    found_zip64 = true;
                    break;
                }
                extra_offset += extra_size;
            }
            if (!found_zip64)
            {
                fprintf(stderr, "zip64 extra field is missing\n");
                return -1;
            }
        }

        if (local_header_offset > archive_size - 4 - sizeof(local_file_header) || file_seek(fp, (int64_t)local_header_offset, SEEK_SET) != 0)
            return -1;

        uint32_t local_signature = 0;
        local_file_header lfh;
        if (fread(&local_signature, sizeof(local_signature), 1, fp) != 1 || local_signature != 0x04034b50 || fread(&lfh, sizeof(lfh), 1, fp) != 1)
        {
            fprintf(stderr, "invalid zip local file header\n");
            return -1;
        }
        if (lfh.compression != cdfh.compression)
        {
            fprintf(stderr, "zip compression method mismatch\n");
            return -1;
        }
        if (cdfh.compression == 0 && compressed_size != uncompressed_size)
        {
            fprintf(stderr, "stored zip record size mismatch\n");
            return -1;
        }

        const uint64_t data_offset = local_header_offset + 4 + sizeof(lfh) + lfh.file_name_length + lfh.extra_field_length;
        if (data_offset > archive_size || compressed_size > archive_size - data_offset)
        {
            fprintf(stderr, "zip file data exceeds archive bounds\n");
            return -1;
        }

        StoreZipMeta meta;
        meta.offset = data_offset;
        meta.size = uncompressed_size;
        meta.compression = cdfh.compression;
        filemetas[name] = meta;

        if (file_seek(fp, next_record, SEEK_SET) != 0)
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

int StoreZipReader::read_file(const std::string& name, char* data)
{
    if (filemetas.find(name) == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return -1;
    }

    uint64_t offset = filemetas[name].offset;
    uint64_t size = filemetas[name].size;
    uint16_t compression = filemetas[name].compression;

    if (compression != 0)
    {
        fprintf(stderr, "compressed zip record is not supported %s\n", name.c_str());
        return -1;
    }

    if (file_seek(fp, (int64_t)offset, SEEK_SET) != 0)
    {
        fprintf(stderr, "seek failed %s\n", name.c_str());
        return -1;
    }

    if (size != 0 && fread(data, size, 1, fp) != 1)
    {
        fprintf(stderr, "read failed %s\n", name.c_str());
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
