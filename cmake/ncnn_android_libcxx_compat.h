// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_ANDROID_LIBCXX_COMPAT_H
#define NCNN_ANDROID_LIBCXX_COMPAT_H

#include <__config>

// NDK r30's libc++ headers otherwise require runtime symbols unavailable in
// older NDKs. Use their header fallbacks, including local stream instantiations.
// These are private libc++ macros; recheck them when updating the NDK/libc++.
#if _LIBCPP_VERSION >= 210000 && _LIBCPP_VERSION < 220000
#undef _LIBCPP_AVAILABILITY_HAS_VERBOSE_ABORT
#define _LIBCPP_AVAILABILITY_HAS_VERBOSE_ABORT 0
#undef _LIBCPP_AVAILABILITY_HAS_HASH_MEMORY
#define _LIBCPP_AVAILABILITY_HAS_HASH_MEMORY 0
#undef _LIBCPP_AVAILABILITY_HAS_ADDITIONAL_IOSTREAM_EXPLICIT_INSTANTIATIONS_1
#define _LIBCPP_AVAILABILITY_HAS_ADDITIONAL_IOSTREAM_EXPLICIT_INSTANTIATIONS_1 0
#endif

#endif // NCNN_ANDROID_LIBCXX_COMPAT_H
