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

// 64-bit file positioning: on Windows long/ftell/fseek stay 32-bit even in a
// 64-bit build, which truncates offsets for archives larger than 2 GiB
#ifdef _MSC_VER
#define PNNX_FSEEK _fseeki64
#define PNNX_FTELL _ftelli64
#else
#define PNNX_FSEEK fseeko
#define PNNX_FTELL ftello
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

int StoreZipReader::open(const std::string& path)
{
    close();

    fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    // locate end of central directory record by scanning backwards
    PNNX_FSEEK(fp, 0, SEEK_END);
    int64_t file_size = PNNX_FTELL(fp);

    uint64_t cd_offset = 0;
    uint64_t cd_size = 0;
    int found = 0;

    // eocd is at most 65557 bytes from the end (22 fixed + up to 65535 comment)
    int64_t minpos = file_size - 65557;
    if (minpos < 0)
        minpos = 0;

    for (int64_t pos = file_size - 22; pos >= minpos; pos--)
    {
        PNNX_FSEEK(fp, pos, SEEK_SET);
        uint32_t signature = 0;
        if (fread((char*)&signature, sizeof(signature), 1, fp) != 1)
            break;

        if (signature != 0x06054b50)
            continue;

        end_of_central_directory_record eocdr;
        fread((char*)&eocdr, sizeof(eocdr), 1, fp);

        // a valid EOCD record plus its comment must extend exactly to the end
        // of the file; otherwise this signature occurrence is inside the zip
        // comment text and must be skipped while continuing the backward scan
        if (pos + 22 + eocdr.comment_length != file_size)
            continue;

        cd_offset = eocdr.cd_offset;
        cd_size = eocdr.cd_size;

        if (eocdr.cd_records == 0xffff || eocdr.total_cd_records == 0xffff || eocdr.cd_offset == 0xffffffff || eocdr.cd_size == 0xffffffff)
        {
            // zip64 : the locator is exactly 20 bytes before the eocd
            PNNX_FSEEK(fp, pos - 20, SEEK_SET);
            uint32_t sig64 = 0;
            fread((char*)&sig64, sizeof(sig64), 1, fp);
            if (sig64 == 0x07064b50)
            {
                zip64_end_of_central_directory_locator eocdl64;
                fread((char*)&eocdl64, sizeof(eocdl64), 1, fp);

                PNNX_FSEEK(fp, (int64_t)eocdl64.eocdr64_offset, SEEK_SET);
                uint32_t sig_eocd64 = 0;
                fread((char*)&sig_eocd64, sizeof(sig_eocd64), 1, fp);
                if (sig_eocd64 == 0x06064b50)
                {
                    zip64_end_of_central_directory_record eocdr64;
                    fread((char*)&eocdr64, sizeof(eocdr64), 1, fp);
                    cd_offset = eocdr64.cd_offset;
                    cd_size = eocdr64.cd_size;
                }
            }
        }

        // the central directory must lie entirely before this EOCD record
        if (cd_offset + cd_size > (uint64_t)pos)
            continue;

        found = 1;
        break;
    }

    if (!found)
    {
        fprintf(stderr, "end of central directory not found\n");
        return -1;
    }

    // walk central directory
    PNNX_FSEEK(fp, (int64_t)cd_offset, SEEK_SET);

    uint64_t pos = cd_offset;
    uint64_t end = cd_offset + cd_size;

    while (pos < end)
    {
        uint32_t signature = 0;
        fread((char*)&signature, sizeof(signature), 1, fp);
        if (signature != 0x02014b50)
        {
            fprintf(stderr, "unsupported central directory signature %x\n", signature);
            return -1;
        }

        central_directory_file_header cdfh;
        fread((char*)&cdfh, sizeof(cdfh), 1, fp);

        std::string name;
        name.resize(cdfh.file_name_length);
        fread((char*)name.data(), name.size(), 1, fp);

        uint64_t compressed_size = cdfh.compressed_size;
        uint64_t uncompressed_size = cdfh.uncompressed_size;
        uint64_t lfh_offset = cdfh.lfh_offset;

        // parse zip64 extended information in the extra field when required
        if (compressed_size == 0xffffffff || uncompressed_size == 0xffffffff || lfh_offset == 0xffffffff)
        {
            uint16_t extra_read = 0;
            while (extra_read < cdfh.extra_field_length)
            {
                uint16_t extra_id = 0;
                uint16_t extra_size = 0;
                fread((char*)&extra_id, sizeof(extra_id), 1, fp);
                fread((char*)&extra_size, sizeof(extra_size), 1, fp);
                extra_read += 4;

                if (extra_id != 0x0001)
                {
                    PNNX_FSEEK(fp, extra_size, SEEK_CUR);
                    extra_read += extra_size;
                    continue;
                }

                if (uncompressed_size == 0xffffffff)
                {
                    fread((char*)&uncompressed_size, sizeof(uncompressed_size), 1, fp);
                    extra_read += 8;
                }
                if (compressed_size == 0xffffffff)
                {
                    fread((char*)&compressed_size, sizeof(compressed_size), 1, fp);
                    extra_read += 8;
                }
                if (lfh_offset == 0xffffffff)
                {
                    fread((char*)&lfh_offset, sizeof(lfh_offset), 1, fp);
                    extra_read += 8;
                }

                // skip remaining bytes of this extra field
                PNNX_FSEEK(fp, cdfh.extra_field_length - extra_read, SEEK_CUR);
                extra_read = cdfh.extra_field_length;
                break;
            }
        }
        else
        {
            // skip extra field
            PNNX_FSEEK(fp, cdfh.extra_field_length, SEEK_CUR);
        }

        // skip file comment
        PNNX_FSEEK(fp, cdfh.file_comment_length, SEEK_CUR);

        if (cdfh.compression != 0 || compressed_size != uncompressed_size)
        {
            fprintf(stderr, "not stored zip file %d %d\n", cdfh.compressed_size, cdfh.uncompressed_size);
            return -1;
        }

        // read local file header to compute the data offset
        // (the local header may carry a data descriptor, sizes there are unreliable)
        int64_t cd_cur = PNNX_FTELL(fp);

        PNNX_FSEEK(fp, (int64_t)lfh_offset, SEEK_SET);
        uint32_t lfh_sig = 0;
        fread((char*)&lfh_sig, sizeof(lfh_sig), 1, fp);
        if (lfh_sig != 0x04034b50)
        {
            fprintf(stderr, "unsupported local header signature %x\n", lfh_sig);
            return -1;
        }

        local_file_header lfh;
        fread((char*)&lfh, sizeof(lfh), 1, fp);

        uint64_t data_offset = lfh_offset + 30 + lfh.file_name_length + lfh.extra_field_length;

        StoreZipMeta fm;
        fm.offset = data_offset;
        fm.size = compressed_size;
        filemetas[name] = fm;

        // back to central directory
        PNNX_FSEEK(fp, cd_cur, SEEK_SET);

        pos += 46 + cdfh.file_name_length + cdfh.extra_field_length + cdfh.file_comment_length;
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

    PNNX_FSEEK(fp, (int64_t)offset, SEEK_SET);
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

int StoreZipWriter::write_file(const std::string& name, const char* data, uint64_t size)
{
    int64_t offset = PNNX_FTELL(fp);

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

    int64_t offset = PNNX_FTELL(fp);

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

    int64_t offset2 = PNNX_FTELL(fp);

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
