// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "storezip.h"

#include "puff.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
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

int StoreZipReader::open(const std::string& path)
{
    close();

    fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    // Read from the central directory. Local headers from torch.export.save
    // set general-purpose bit 3 (data descriptor) and leave sizes as 0.
    fseek(fp, 0, SEEK_END);
    long filesize_l = ftell(fp);
    if (filesize_l < 22)
    {
        fprintf(stderr, "zip too small\n");
        return -1;
    }
    uint64_t filesize = (uint64_t)filesize_l;

    const uint64_t max_comment = 65535;
    uint64_t search_len = filesize < (max_comment + 22) ? filesize : (max_comment + 22);
    std::vector<unsigned char> tail(search_len);
    fseek(fp, (long)(filesize - search_len), SEEK_SET);
    if (fread(tail.data(), search_len, 1, fp) != 1)
    {
        fprintf(stderr, "zip read tail failed\n");
        return -1;
    }

    int eocd_rel = -1;
    for (int i = (int)search_len - 22; i >= 0; i--)
    {
        if (tail[i] != 0x50 || tail[i + 1] != 0x4b || tail[i + 2] != 0x05 || tail[i + 3] != 0x06)
            continue;
        uint16_t comment_length = 0;
        memcpy(&comment_length, &tail[i + 20], 2);
        if ((uint64_t)i + 22 + comment_length == search_len)
        {
            eocd_rel = i;
            break;
        }
    }
    if (eocd_rel < 0)
    {
        fprintf(stderr, "zip end of central directory not found\n");
        return -1;
    }

    uint16_t total_cd_records16 = 0;
    uint32_t cd_size32 = 0;
    uint32_t cd_offset32 = 0;
    memcpy(&total_cd_records16, &tail[eocd_rel + 10], 2);
    memcpy(&cd_size32, &tail[eocd_rel + 12], 4);
    memcpy(&cd_offset32, &tail[eocd_rel + 16], 4);

    uint64_t cd_offset = cd_offset32;
    uint64_t cd_records = total_cd_records16;

    if (cd_offset32 == 0xffffffff || total_cd_records16 == 0xffff)
    {
        if (eocd_rel < 20)
        {
            fprintf(stderr, "zip64 locator missing\n");
            return -1;
        }
        const unsigned char* loc = &tail[eocd_rel - 20];
        uint32_t loc_sig = 0;
        memcpy(&loc_sig, loc, 4);
        if (loc_sig != 0x07064b50)
        {
            fprintf(stderr, "zip64 locator signature mismatch %x\n", loc_sig);
            return -1;
        }
        uint64_t eocdr64_offset = 0;
        memcpy(&eocdr64_offset, loc + 8, 8);

        fseek(fp, (long)eocdr64_offset, SEEK_SET);
        uint32_t zsig = 0;
        if (fread((char*)&zsig, sizeof(zsig), 1, fp) != 1 || zsig != 0x06064b50)
        {
            fprintf(stderr, "zip64 eocd signature mismatch\n");
            return -1;
        }
        zip64_end_of_central_directory_record eocdr64;
        if (fread((char*)&eocdr64, sizeof(eocdr64), 1, fp) != 1)
        {
            fprintf(stderr, "zip64 eocd read failed\n");
            return -1;
        }
        cd_records = eocdr64.total_cd_records;
        cd_offset = eocdr64.cd_offset;
        (void)cd_size32;
    }

    fseek(fp, (long)cd_offset, SEEK_SET);
    for (uint64_t i = 0; i < cd_records; i++)
    {
        uint32_t signature = 0;
        if (fread((char*)&signature, sizeof(signature), 1, fp) != 1 || signature != 0x02014b50)
        {
            fprintf(stderr, "zip central directory signature mismatch %x\n", signature);
            return -1;
        }

        central_directory_file_header cdfh;
        if (fread((char*)&cdfh, sizeof(cdfh), 1, fp) != 1)
        {
            fprintf(stderr, "zip central directory header read failed\n");
            return -1;
        }

        if (cdfh.compression != 0 && cdfh.compression != 8)
        {
            fprintf(stderr, "unsupported zip compression %d\n", cdfh.compression);
            return -1;
        }

        std::string name;
        name.resize(cdfh.file_name_length);
        if (cdfh.file_name_length && fread((char*)name.data(), name.size(), 1, fp) != 1)
        {
            fprintf(stderr, "zip filename read failed\n");
            return -1;
        }

        std::vector<unsigned char> extra(cdfh.extra_field_length);
        if (cdfh.extra_field_length && fread(extra.data(), extra.size(), 1, fp) != 1)
        {
            fprintf(stderr, "zip extra field read failed\n");
            return -1;
        }

        if (cdfh.file_comment_length)
            fseek(fp, cdfh.file_comment_length, SEEK_CUR);

        uint64_t compressed_size = cdfh.compressed_size;
        uint64_t uncompressed_size = cdfh.uncompressed_size;
        uint64_t lfh_offset = cdfh.lfh_offset;

        size_t extra_off = 0;
        while (extra_off + 4 <= extra.size())
        {
            uint16_t extra_id = 0;
            uint16_t extra_size = 0;
            memcpy(&extra_id, extra.data() + extra_off, 2);
            memcpy(&extra_size, extra.data() + extra_off + 2, 2);
            extra_off += 4;
            if (extra_off + extra_size > extra.size())
                break;
            if (extra_id == 0x0001)
            {
                const unsigned char* p = extra.data() + extra_off;
                size_t po = 0;
                if (cdfh.uncompressed_size == 0xffffffff && po + 8 <= extra_size)
                {
                    memcpy(&uncompressed_size, p + po, 8);
                    po += 8;
                }
                if (cdfh.compressed_size == 0xffffffff && po + 8 <= extra_size)
                {
                    memcpy(&compressed_size, p + po, 8);
                    po += 8;
                }
                if (cdfh.lfh_offset == 0xffffffff && po + 8 <= extra_size)
                {
                    memcpy(&lfh_offset, p + po, 8);
                    po += 8;
                }
            }
            extra_off += extra_size;
        }

        long cd_pos = ftell(fp);
        fseek(fp, (long)lfh_offset, SEEK_SET);
        uint32_t lsig = 0;
        if (fread((char*)&lsig, sizeof(lsig), 1, fp) != 1 || lsig != 0x04034b50)
        {
            fprintf(stderr, "zip local header signature mismatch %x\n", lsig);
            return -1;
        }
        local_file_header lfh;
        if (fread((char*)&lfh, sizeof(lfh), 1, fp) != 1)
        {
            fprintf(stderr, "zip local header read failed\n");
            return -1;
        }
        fseek(fp, lfh.file_name_length + lfh.extra_field_length, SEEK_CUR);

        StoreZipMeta fm;
        fm.offset = (uint64_t)ftell(fp);
        fm.compressed_size = compressed_size;
        fm.uncompressed_size = uncompressed_size;
        fm.compression = cdfh.compression;
        filemetas[name] = fm;

        fseek(fp, cd_pos, SEEK_SET);
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

    return filemetas.at(name).uncompressed_size;
}

int StoreZipReader::read_file(const std::string& name, char* data)
{
    if (filemetas.find(name) == filemetas.end())
    {
        fprintf(stderr, "no such file %s\n", name.c_str());
        return -1;
    }

    const StoreZipMeta& fm = filemetas[name];

    fseek(fp, fm.offset, SEEK_SET);

    if (fm.compression == 0)
    {
        if (fm.uncompressed_size > 0)
            fread(data, fm.uncompressed_size, 1, fp);
        return 0;
    }

    if (fm.compression != 8)
    {
        fprintf(stderr, "unsupported zip compression %d for %s\n", fm.compression, name.c_str());
        return -1;
    }

    if (fm.compressed_size == 0)
        return 0;

    std::vector<unsigned char> src(fm.compressed_size);
    fread(src.data(), fm.compressed_size, 1, fp);

    unsigned long destlen = (unsigned long)fm.uncompressed_size;
    unsigned long sourcelen = (unsigned long)fm.compressed_size;
    int ret = puff((unsigned char*)data, &destlen, src.data(), &sourcelen);
    if (ret != 0)
    {
        fprintf(stderr, "puff inflate failed %d for %s\n", ret, name.c_str());
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
