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

// minimal raw-DEFLATE (RFC 1951) inflate for compressed zip entries.
// zip method 8 stores a raw deflate stream (no zlib wrapper). supports the
// stored / fixed-huffman / dynamic-huffman blocks that python/torch/zip emit.
// zero third-party dependency, mirroring the rest of the storezip reader.
struct DeflateBitReader
{
    const unsigned char* in;
    size_t in_len;
    size_t in_pos;
    unsigned int bitbuf;
    int bitcnt;

    DeflateBitReader(const unsigned char* i, size_t l)
    {
        in = i;
        in_len = l;
        in_pos = 0;
        bitbuf = 0;
        bitcnt = 0;
    }

    int getbit(int& bit)
    {
        if (bitcnt == 0)
        {
            if (in_pos >= in_len)
                return -1;
            bitbuf = in[in_pos++];
            bitcnt = 8;
        }
        bit = (int)(bitbuf & 1);
        bitbuf >>= 1;
        bitcnt--;
        return 0;
    }

    int getbits(int n, int& val)
    {
        val = 0;
        for (int i = 0; i < n; i++)
        {
            int b;
            if (getbit(b))
                return -1;
            val |= b << i;
        }
        return 0;
    }
};

struct DeflateHuff
{
    int count[16];   // count[code_length]
    int symbol[288]; // symbols in canonical order (length asc, then value asc)
    int maxbits;
};

static void deflate_build_huff(const int* lengths, int num_symbols, DeflateHuff& h)
{
    for (int i = 0; i < 16; i++)
        h.count[i] = 0;
    h.maxbits = 0;

    for (int s = 0; s < num_symbols; s++)
    {
        const int len = lengths[s];
        if (len > 0)
        {
            h.count[len]++;
            if (len > h.maxbits)
                h.maxbits = len;
        }
    }

    int idx = 0;
    for (int len = 1; len <= h.maxbits; len++)
    {
        for (int s = 0; s < num_symbols; s++)
        {
            if (lengths[s] == len)
                h.symbol[idx++] = s;
        }
    }
}

static int deflate_decode_symbol(DeflateBitReader& br, const DeflateHuff& h, int& symbol)
{
    int code = 0;
    int first = 0;
    int index = 0;

    for (int len = 1; len <= h.maxbits; len++)
    {
        int b;
        if (br.getbit(b))
            return -1;
        code |= b;

        const int count = h.count[len];
        if (code - count < first)
        {
            symbol = h.symbol[index + (code - first)];
            return 0;
        }

        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }

    return -1;
}

// length code 257..285 -> (base, extra bits)
static const unsigned short DEFLATE_LEN_BASE[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
};
static const unsigned char DEFLATE_LEN_EXTRA[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
};

// distance code 0..29 -> (base, extra bits)
static const unsigned short DEFLATE_DIST_BASE[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
};
static const unsigned char DEFLATE_DIST_EXTRA[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
};

static int deflate_inflate_stream(DeflateBitReader& br, unsigned char* out, size_t out_cap, size_t& out_pos)
{
    // code-length-code alphabet order (RFC 1951)
    static const int CLCLS[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

    int lengths_lit[288] = {0};
    int lengths_dist[30] = {0};

    for (;;)
    {
        int bfinal = 0;
        int btype = 0;
        if (br.getbits(3, btype))
            return -1;
        bfinal = btype & 1;
        btype >>= 1;

        if (btype == 0)
        {
            // stored block: skip to byte boundary, then LEN/NLEN
            br.bitbuf = 0;
            br.bitcnt = 0;

            int len = 0;
            int nlen = 0;
            if (br.getbits(16, len))
                return -1;
            if (br.getbits(16, nlen))
                return -1;
            if (len != (~nlen & 0xffff))
                return -1;

            if (br.in_pos + (size_t)len > br.in_len)
                return -1;
            if (out_pos + (size_t)len > out_cap)
                return -1;

            for (int i = 0; i < len; i++)
                out[out_pos++] = br.in[br.in_pos++];
        }
        else if (btype == 1)
        {
            // fixed huffman
            for (int s = 0; s < 144; s++)
                lengths_lit[s] = 8;
            for (int s = 144; s < 256; s++)
                lengths_lit[s] = 9;
            for (int s = 256; s < 280; s++)
                lengths_lit[s] = 7;
            for (int s = 280; s < 288; s++)
                lengths_lit[s] = 8;
            for (int s = 0; s < 30; s++)
                lengths_dist[s] = 5;

            DeflateHuff hlit;
            DeflateHuff hdist;
            deflate_build_huff(lengths_lit, 288, hlit);
            deflate_build_huff(lengths_dist, 30, hdist);

            // decode literals / matches
            for (;;)
            {
                int sym = 0;
                if (deflate_decode_symbol(br, hlit, sym))
                    return -1;

                if (sym < 256)
                {
                    if (out_pos >= out_cap)
                        return -1;
                    out[out_pos++] = (unsigned char)sym;
                }
                else if (sym == 256)
                {
                    break; // end of block
                }
                else
                {
                    const int li = sym - 257;
                    if (li < 0 || li >= 29)
                        return -1;
                    int length = DEFLATE_LEN_BASE[li];
                    int lextra = DEFLATE_LEN_EXTRA[li];
                    if (lextra > 0)
                    {
                        int v = 0;
                        if (br.getbits(lextra, v))
                            return -1;
                        length += v;
                    }

                    int dsym = 0;
                    if (deflate_decode_symbol(br, hdist, dsym))
                        return -1;
                    if (dsym < 0 || dsym >= 30)
                        return -1;
                    int distance = DEFLATE_DIST_BASE[dsym];
                    int dextra = DEFLATE_DIST_EXTRA[dsym];
                    if (dextra > 0)
                    {
                        int v = 0;
                        if (br.getbits(dextra, v))
                            return -1;
                        distance += v;
                    }

                    if ((size_t)distance > out_pos)
                        return -1;
                    if (out_pos + (size_t)length > out_cap)
                        return -1;

                    for (int i = 0; i < length; i++)
                    {
                        out[out_pos] = out[out_pos - (size_t)distance];
                        out_pos++;
                    }
                }
            }
        }
        else if (btype == 2)
        {
            // dynamic huffman
            int hlit = 0;
            int hdist = 0;
            int hclen = 0;
            if (br.getbits(5, hlit))
                return -1;
            if (br.getbits(5, hdist))
                return -1;
            if (br.getbits(4, hclen))
                return -1;
            hlit += 257;
            hdist += 1;
            hclen += 4;

            // RFC1951 caps: HLIT in 257..286, HDIST in 1..30. a hostile header
            // (HLIT=288, HDIST=32) would overflow all_lengths below, so reject
            // it before decoding instead of writing past the stack array
            if (hlit > 286 || hdist > 30)
                return -1;

            int lengths_cl[19] = {0};
            for (int i = 0; i < hclen; i++)
            {
                int v = 0;
                if (br.getbits(3, v))
                    return -1;
                lengths_cl[CLCLS[i]] = v;
            }

            DeflateHuff hcl;
            deflate_build_huff(lengths_cl, 19, hcl);

            // decode the lit/dist code lengths (with run codes 16/17/18)
            int all_lengths[288 + 30];
            int total = hlit + hdist;
            int idx = 0;
            int prev = 0;
            while (idx < total)
            {
                int sym = 0;
                if (deflate_decode_symbol(br, hcl, sym))
                    return -1;

                if (sym < 16)
                {
                    all_lengths[idx++] = sym;
                    prev = sym;
                }
                else if (sym == 16)
                {
                    int rep = 0;
                    if (br.getbits(2, rep))
                        return -1;
                    rep += 3;
                    if (idx + rep > total)
                        return -1;
                    for (int i = 0; i < rep; i++)
                        all_lengths[idx++] = prev;
                }
                else if (sym == 17)
                {
                    int rep = 0;
                    if (br.getbits(3, rep))
                        return -1;
                    rep += 3;
                    if (idx + rep > total)
                        return -1;
                    for (int i = 0; i < rep; i++)
                        all_lengths[idx++] = 0;
                    prev = 0;
                }
                else // sym == 18
                {
                    int rep = 0;
                    if (br.getbits(7, rep))
                        return -1;
                    rep += 11;
                    if (idx + rep > total)
                        return -1;
                    for (int i = 0; i < rep; i++)
                        all_lengths[idx++] = 0;
                    prev = 0;
                }
            }

            for (int i = 0; i < hlit; i++)
                lengths_lit[i] = all_lengths[i];
            for (int i = 0; i < hdist; i++)
                lengths_dist[i] = all_lengths[hlit + i];

            DeflateHuff hlit2;
            DeflateHuff hdist2;
            deflate_build_huff(lengths_lit, 288, hlit2);
            deflate_build_huff(lengths_dist, 30, hdist2);

            // decode literals / matches (same as fixed block)
            for (;;)
            {
                int sym = 0;
                if (deflate_decode_symbol(br, hlit2, sym))
                    return -1;

                if (sym < 256)
                {
                    if (out_pos >= out_cap)
                        return -1;
                    out[out_pos++] = (unsigned char)sym;
                }
                else if (sym == 256)
                {
                    break;
                }
                else
                {
                    const int li = sym - 257;
                    if (li < 0 || li >= 29)
                        return -1;
                    int length = DEFLATE_LEN_BASE[li];
                    const int lextra = DEFLATE_LEN_EXTRA[li];
                    if (lextra > 0)
                    {
                        int v = 0;
                        if (br.getbits(lextra, v))
                            return -1;
                        length += v;
                    }

                    int dsym = 0;
                    if (deflate_decode_symbol(br, hdist2, dsym))
                        return -1;
                    if (dsym < 0 || dsym >= 30)
                        return -1;
                    int distance = DEFLATE_DIST_BASE[dsym];
                    const int dextra = DEFLATE_DIST_EXTRA[dsym];
                    if (dextra > 0)
                    {
                        int v = 0;
                        if (br.getbits(dextra, v))
                            return -1;
                        distance += v;
                    }

                    if ((size_t)distance > out_pos)
                        return -1;
                    if (out_pos + (size_t)length > out_cap)
                        return -1;

                    for (int i = 0; i < length; i++)
                    {
                        out[out_pos] = out[out_pos - (size_t)distance];
                        out_pos++;
                    }
                }
            }
        }
        else
        {
            return -1; // reserved block type
        }

        if (bfinal)
            return 0;
    }
}

static int deflate_inflate(const unsigned char* comp, size_t comp_len, unsigned char* out, size_t out_cap, size_t& out_pos)
{
    DeflateBitReader br(comp, comp_len);
    out_pos = 0;
    return deflate_inflate_stream(br, out, out_cap, out_pos);
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

    // CRC32_TABLE is shared with the writer and only initialized there; the
    // reader now verifies entry crc32s, so make sure the table exists
    CRC32_TABLE_INIT();

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

        if (cdfh.compression != 0 && cdfh.compression != 8)
        {
            fprintf(stderr, "unsupported zip compression method %d\n", cdfh.compression);
            return -1;
        }

        if (cdfh.flag & 1)
        {
            fprintf(stderr, "encrypted zip entry %s\n", name.c_str());
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
        fm.uncompressed_size = uncompressed_size;
        fm.compression = cdfh.compression;
        fm.crc32 = cdfh.crc32;
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

    return filemetas.at(name).uncompressed_size;
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
    uint64_t uncompressed_size = filemetas[name].uncompressed_size;
    uint16_t compression = filemetas[name].compression;

    PNNX_FSEEK(fp, (int64_t)offset, SEEK_SET);

    if (compression == 8)
    {
        std::vector<char> comp((size_t)size);
        fread(comp.data(), size, 1, fp);

        size_t out_pos = 0;
        if (deflate_inflate((const unsigned char*)comp.data(), (size_t)size, (unsigned char*)data, (size_t)uncompressed_size, out_pos) != 0)
        {
            fprintf(stderr, "inflate failed %s\n", name.c_str());
            return -1;
        }
        if (out_pos != (size_t)uncompressed_size)
        {
            fprintf(stderr, "inflate size mismatch %s %lu %lu\n", name.c_str(), (unsigned long)out_pos, (unsigned long)uncompressed_size);
            return -1;
        }
    }
    else
    {
        fread(data, size, 1, fp);
    }

    // data integrity: the central directory carries the crc32 of the
    // uncompressed entry; reject silently-corrupted payloads here instead of
    // materializing wrong weights/constants downstream
    if (CRC32_buffer((const unsigned char*)data, uncompressed_size) != filemetas[name].crc32)
    {
        fprintf(stderr, "crc mismatch %s\n", name.c_str());
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
