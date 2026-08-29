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

// Little-endian readers that are safe against misalignment and endianness.
static uint16_t read_le16(const unsigned char* p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t read_le32(const unsigned char* p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t read_le64(const unsigned char* p)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; i++)
        v |= (uint64_t)p[i] << (8 * i);
    return v;
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

    // Locate the central directory via the End Of Central Directory (EOCD)
    // record. The central directory is the authoritative index of a zip and is
    // not affected by data descriptors (general purpose bit 3, 0x08) that some
    // writers (e.g. PyTorch's torch.export.save) put on local file headers.
    // Reading through the central directory makes this reader robust to data
    // descriptors instead of rejecting them.
    uint64_t cd_offset = 0;
    uint64_t cd_size = 0;
    uint64_t cd_records = 0;

    if (find_central_directory(cd_offset, cd_size, cd_records) != 0)
    {
        fprintf(stderr, "store zip: failed to locate central directory\n");
        close();
        return -1;
    }

    // Phase 1: read every central directory file header into a temporary list.
    // The central directory is the authoritative index: it carries the true
    // compressed size (even for entries written with a data descriptor) and the
    // local-header offset.
    if (fseek(fp, cd_offset, SEEK_SET) != 0)
    {
        fprintf(stderr, "store zip: seek to central directory failed\n");
        close();
        return -1;
    }

    struct CDEntry
    {
        std::string name;
        uint64_t compressed_size;
        uint64_t lfh_offset;
    };
    std::vector<CDEntry> cdentries;
    cdentries.reserve((size_t)cd_records);

    for (uint64_t i = 0; i < cd_records; i++)
    {
        uint32_t signature;
        if (fread((char*)&signature, sizeof(signature), 1, fp) != 1)
            break;

        if (signature != 0x02014b50)
        {
            fprintf(stderr, "store zip: bad central directory signature %x\n", signature);
            close();
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

        // Resolve zip64 extended information from the central directory extra
        // field when the 32-bit fields are saturated to 0xffffffff.
        if (compressed_size == 0xffffffff || uncompressed_size == 0xffffffff || lfh_offset == 0xffffffff)
        {
            uint16_t extra_offset = 0;
            while (extra_offset + 4 <= cdfh.extra_field_length)
            {
                uint16_t extra_id;
                uint16_t extra_size;
                fread((char*)&extra_id, sizeof(extra_id), 1, fp);
                fread((char*)&extra_size, sizeof(extra_size), 1, fp);
                if (extra_id == 0x0001)
                {
                    zip64_extended_extra_field zip64_eef;
                    fread((char*)&zip64_eef, sizeof(zip64_eef), 1, fp);
                    if (uncompressed_size == 0xffffffff)
                        uncompressed_size = zip64_eef.uncompressed_size;
                    if (compressed_size == 0xffffffff)
                        compressed_size = zip64_eef.compressed_size;
                    if (lfh_offset == 0xffffffff)
                        lfh_offset = zip64_eef.lfh_offset;
                    fseek(fp, cdfh.extra_field_length - extra_offset - 4 - sizeof(zip64_eef), SEEK_CUR);
                    break;
                }
                else
                {
                    fseek(fp, extra_size - 4, SEEK_CUR);
                    extra_offset += extra_size;
                }
            }
        }
        else
        {
            fseek(fp, cdfh.extra_field_length, SEEK_CUR);
        }

        // skip file comment
        fseek(fp, cdfh.file_comment_length, SEEK_CUR);

        if (cdfh.compression != 0 || compressed_size != uncompressed_size)
        {
            fprintf(stderr, "not stored zip file %d %d\n", (int)compressed_size, (int)uncompressed_size);
            close();
            return -1;
        }

        CDEntry e;
        e.name = name;
        e.compressed_size = compressed_size;
        e.lfh_offset = lfh_offset;
        cdentries.push_back(e);
    }

    // Phase 2: for each entry, seek to its local file header to compute the real
    // data offset (after the local file name + extra field). This is done as a
    // separate pass so seeking to the local header does not disturb the central
    // directory scan above.
    for (size_t i = 0; i < cdentries.size(); i++)
    {
        const CDEntry& e = cdentries[i];

        if (fseek(fp, e.lfh_offset, SEEK_SET) != 0)
        {
            fprintf(stderr, "store zip: seek to local header failed for %s\n", e.name.c_str());
            close();
            return -1;
        }

        uint32_t lfh_signature;
        fread((char*)&lfh_signature, sizeof(lfh_signature), 1, fp);
        if (lfh_signature != 0x04034b50)
        {
            fprintf(stderr, "store zip: bad local header signature for %s\n", e.name.c_str());
            close();
            return -1;
        }

        local_file_header lfh;
        fread((char*)&lfh, sizeof(lfh), 1, fp);

        // skip file name + extra field of the local header to reach the data
        fseek(fp, lfh.file_name_length + lfh.extra_field_length, SEEK_CUR);

        StoreZipMeta fm;
        fm.offset = ftell(fp);
        fm.size = e.compressed_size;

        filemetas[e.name] = fm;
    }

    return 0;
}

int StoreZipReader::find_central_directory(uint64_t& cd_offset, uint64_t& cd_size, uint64_t& cd_records)
{
    // Determine file size.
    if (fseek(fp, 0, SEEK_END) != 0)
        return -1;
    long file_size = ftell(fp);
    if (file_size < 22)
        return -1;

    // The EOCD record sits at the very end of the file, optionally followed by a
    // comment of at most 65535 bytes. Scan backward for its signature.
    long scan_start = file_size - 22 - 65535;
    if (scan_start < 0)
        scan_start = 0;

    std::vector<unsigned char> buf(file_size - scan_start);
    if (fseek(fp, scan_start, SEEK_SET) != 0)
        return -1;
    if (fread(buf.data(), buf.size(), 1, fp) != 1)
        return -1;

    // Scan backward for EOCD candidates. Each candidate is validated by checking
    // that cd_offset really points at a central directory file header; this
    // rejects spurious 50 4b 05 06 byte sequences that may appear in trailing
    // comment bytes or elsewhere in the payload (such as a PK\x05\x06 that some
    // serializers embed).
    for (long p = (long)buf.size() - 22; p >= 0; p--)
    {
        if (!(buf[p] == 0x50 && buf[p + 1] == 0x4b && buf[p + 2] == 0x05 && buf[p + 3] == 0x06))
            continue;

        long eocd_buf_off = p;
        uint16_t eocd_records = read_le16(buf.data() + eocd_buf_off + 10);
        uint32_t eocd_cd_size = read_le32(buf.data() + eocd_buf_off + 12);
        uint32_t eocd_cd_offset = read_le32(buf.data() + eocd_buf_off + 16);

        if (eocd_cd_offset != 0xffffffff)
        {
            if (cd_offset_valid(eocd_cd_offset) && (uint64_t)eocd_cd_offset + eocd_cd_size <= (uint64_t)file_size)
            {
                cd_offset = eocd_cd_offset;
                cd_size = eocd_cd_size;
                cd_records = eocd_records;
                return 0;
            }
            continue;
        }

        // Zip64: the zip64 EOCD locator is 20 bytes before this EOCD candidate.
        long loc_pos = (scan_start + p) - 20;
        if (loc_pos < 0)
            continue;
        long lp = loc_pos - scan_start;
        if (!(buf[lp] == 0x50 && buf[lp + 1] == 0x4b && buf[lp + 2] == 0x06 && buf[lp + 3] == 0x07))
            continue;
        uint64_t eocdr64_offset = read_le64(buf.data() + lp + 8);

        // Read + validate the zip64 EOCD record.
        if (fseek(fp, (long)eocdr64_offset, SEEK_SET) != 0)
            continue;
        uint32_t sig;
        if (fread((char*)&sig, sizeof(sig), 1, fp) != 1)
            continue;
        if (sig != 0x06064b50)
            continue;
        // skip size_of_eocd64_m12 (8), version_made_by (2), version_min_required (2),
        // disk_number (4), start_disk (4)
        if (fseek(fp, 8 + 2 + 2 + 4 + 4, SEEK_CUR) != 0)
            continue;
        uint64_t z64_cd_records;
        uint64_t z64_cd_size;
        uint64_t z64_cd_offset;
        if (fread((char*)&z64_cd_records, sizeof(z64_cd_records), 1, fp) != 1)
            continue;
        if (fread((char*)&z64_cd_size, sizeof(z64_cd_size), 1, fp) != 1)
            continue;
        if (fread((char*)&z64_cd_offset, sizeof(z64_cd_offset), 1, fp) != 1)
            continue;

        if (cd_offset_valid(z64_cd_offset) && z64_cd_offset + z64_cd_size <= (uint64_t)file_size)
        {
            cd_offset = z64_cd_offset;
            cd_size = z64_cd_size;
            cd_records = z64_cd_records;
            return 0;
        }
    }

    return -1;
}

bool StoreZipReader::cd_offset_valid(uint64_t off)
{
    if (off == 0)
        return false;
    if (fseek(fp, (long)off, SEEK_SET) != 0)
        return false;
    uint32_t sig;
    if (fread((char*)&sig, sizeof(sig), 1, fp) != 1)
        return false;
    return sig == 0x02014b50;
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
