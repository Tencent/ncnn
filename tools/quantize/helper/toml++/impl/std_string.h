//# This file is a part of toml++ and is subject to the the terms of the MIT license.
//# Copyright (c) Mark Gillard <mark.gillard@outlook.com.au>
//# See https://github.com/marzer/tomlplusplus/blob/master/LICENSE for the full license text.
// SPDX-License-Identifier: MIT
#pragma once

#include "preprocessor.h"
TOML_DISABLE_WARNINGS;
#include <string_view>
#include <string>
TOML_ENABLE_WARNINGS;

#if defined(DOXYGEN)                                                                                                   \
	|| (defined(__cpp_char8_t) && __cpp_char8_t >= 201811 && defined(__cpp_lib_char8_t)                                \
		&& __cpp_lib_char8_t >= 201907)
#define TOML_HAS_CHAR8 1
#else
#define TOML_HAS_CHAR8 0
#endif

/// \cond

namespace toml // non-abi namespace; this is not an error
{
	using namespace std::string_literals;
	using namespace std::string_view_literals;
}

#if TOML_ENABLE_WINDOWS_COMPAT

TOML_IMPL_NAMESPACE_START
{
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	std::string narrow(std::wstring_view);

	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	std::wstring widen(std::string_view);

#if TOML_HAS_CHAR8

	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	std::wstring widen(std::u8string_view);

#endif
}
TOML_IMPL_NAMESPACE_END;

#endif // TOML_ENABLE_WINDOWS_COMPAT

/// \endcond
