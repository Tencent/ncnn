// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_TOOLS_FILEIO_H
#define NCNN_TOOLS_FILEIO_H

#include <stdint.h>
#include <stdio.h>

#if !defined(_WIN32)
#include <sys/types.h>
#endif

// The Windows CRT keeps long at 32 bits even in 64-bit builds.  Keep tool
// file positions independent of long so model and archive files can exceed 2 GiB.
static inline int64_t ncnn_file_tell(FILE* fp)
{
#if defined(_WIN32)
    return (int64_t)_ftelli64(fp);
#else
    return (int64_t)ftello(fp);
#endif
}

static inline int ncnn_file_seek(FILE* fp, int64_t offset, int origin)
{
#if defined(_WIN32)
    return _fseeki64(fp, (__int64)offset, origin);
#else
    return fseeko(fp, (off_t)offset, origin);
#endif
}

#endif // NCNN_TOOLS_FILEIO_H
