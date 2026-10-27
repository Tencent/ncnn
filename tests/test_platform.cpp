// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "platform.h"

#include <string.h>

#if NCNN_STDIO && (!defined(__ANDROID__) || __ANDROID_API__ >= 21) && ((defined(_WIN32) && !NCNN_WINXP) || defined(__ANDROID__) || defined(__OHOS__) || defined(__linux__) || __APPLE__)

// An explicit test-local directory avoids depending on a writable system TMPDIR.
class TemporaryTestDirectory
{
public:
    TemporaryTestDirectory()
    {
        path[0] = 0;
    }
    ~TemporaryTestDirectory()
    {
        remove();
    }

    int create()
    {
#if defined(_WIN32)
        const unsigned long process = GetCurrentProcessId();
#else
        const unsigned long process = (unsigned long)getpid();
#endif
        for (int i = 0; i < 64; i++)
        {
            snprintf(path, sizeof(path), "./ncnn-tempfile-test-%lu-%d", process, i);
#if defined(_WIN32)
            if (CreateDirectoryA(path, 0))
                return 0;
            if (GetLastError() != ERROR_ALREADY_EXISTS)
                break;
#else
            if (mkdir(path, 0700) == 0)
                return 0;
            if (errno != EEXIST)
                break;
#endif
        }
        path[0] = 0;
        return -1;
    }

    int remove()
    {
        if (!path[0])
            return 0;
#if defined(_WIN32)
        if (!RemoveDirectoryA(path))
            return -1;
#else
        if (rmdir(path) != 0)
            return -1;
#endif
        path[0] = 0;
        return 0;
    }

    char path[128];
};

class TemporaryTestMapping : public ncnn::TemporaryFile::Mapping
{
public:
    ~TemporaryTestMapping()
    {
        ncnn::TemporaryFile::unmap(*this);
    }
};

#define CHECK_TEMPORARY_FILE(condition)                                   \
    do                                                                    \
    {                                                                     \
        if (!(condition))                                                 \
        {                                                                 \
            NCNN_LOGE("test_temporary_file failed at line %d", __LINE__); \
            return -1;                                                    \
        }                                                                 \
    } while (0)

static int test_temporary_file()
{
    TemporaryTestDirectory directory;
    CHECK_TEMPORARY_FILE(directory.create() == 0);
    {
        ncnn::TemporaryFile file;
        TemporaryTestMapping first;
        TemporaryTestMapping second;
        TemporaryTestMapping invalid;
        CHECK_TEMPORARY_FILE(!file.is_open());

        char missing[160];
        snprintf(missing, sizeof(missing), "%s/missing/directory", directory.path);
        CHECK_TEMPORARY_FILE(file.open(missing) != 0);
        CHECK_TEMPORARY_FILE(!file.is_open());
        CHECK_TEMPORARY_FILE(file.reserve(16) != 0);
        CHECK_TEMPORARY_FILE(file.map(0, 16, invalid) != 0 && !invalid.base);

        CHECK_TEMPORARY_FILE(file.open(directory.path) == 0);
        CHECK_TEMPORARY_FILE(file.is_open());
        CHECK_TEMPORARY_FILE(file.reserve(4097) == 0);
        CHECK_TEMPORARY_FILE(file.map(3, 37, first) == 0);
        memset(first.data, 0x31, 37);

        // Growing and mapping another range must not invalidate the first view.
        CHECK_TEMPORARY_FILE(file.reserve(16384) == 0);
        CHECK_TEMPORARY_FILE(file.map(8195, 123, second) == 0);
        memset(second.data, 0x32, 123);
        CHECK_TEMPORARY_FILE(((const unsigned char*)first.data)[36] == 0x31);
        CHECK_TEMPORARY_FILE(file.reserve(8) == 0);
        CHECK_TEMPORARY_FILE(file.map(16383, 2, invalid) != 0 && !invalid.base);
        CHECK_TEMPORARY_FILE(file.map(0, 0, invalid) != 0 && !invalid.base);
        CHECK_TEMPORARY_FILE(file.reserve((uint64_t)-1) != 0);

        CHECK_TEMPORARY_FILE(ncnn::TemporaryFile::flush_readonly(first) == 0);
        CHECK_TEMPORARY_FILE(ncnn::TemporaryFile::flush_readonly(second) == 0);
        CHECK_TEMPORARY_FILE(((const unsigned char*)first.data)[0] == 0x31);
        CHECK_TEMPORARY_FILE(((const unsigned char*)second.data)[122] == 0x32);
        ncnn::TemporaryFile::unmap(first);
        ncnn::TemporaryFile::unmap(second);
        CHECK_TEMPORARY_FILE(!first.base && !first.data && !first.mapped_size);
        CHECK_TEMPORARY_FILE(!second.base && !second.data && !second.mapped_size);
        file.close();
        file.close();
        CHECK_TEMPORARY_FILE(!file.is_open());

        CHECK_TEMPORARY_FILE(file.open(directory.path) == 0);
        CHECK_TEMPORARY_FILE(file.reserve(16) == 0);
        CHECK_TEMPORARY_FILE(file.map(0, 16, first) == 0);
        memset(first.data, 0x33, 16);
        // The mapping and file destructors clean up this second opening.
    }
    // Removing the directory also verifies that no temporary filename leaked.
    CHECK_TEMPORARY_FILE(directory.remove() == 0);
    return 0;
}

#undef CHECK_TEMPORARY_FILE
#endif

int main()
{
#if NCNN_STDIO && (!defined(__ANDROID__) || __ANDROID_API__ >= 21) && ((defined(_WIN32) && !NCNN_WINXP) || defined(__ANDROID__) || defined(__OHOS__) || defined(__linux__) || __APPLE__)
    return test_temporary_file();
#else
    return 0;
#endif
}
