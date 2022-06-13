//# This file is a part of toml++ and is subject to the the terms of the MIT license.
//# Copyright (c) Mark Gillard <mark.gillard@outlook.com.au>
//# See https://github.com/marzer/tomlplusplus/blob/master/LICENSE for the full license text.
// SPDX-License-Identifier: MIT
#pragma once

#include "preprocessor.h"
#if TOML_ENABLE_PARSER

#include "table.h"
#include "parse_result.h"
#include "header_start.h"

TOML_NAMESPACE_START
{
	TOML_ABI_NAMESPACE_BOOL(TOML_EXCEPTIONS, ex, noex);

	/// \brief	Parses a TOML document from a string view.
	///
	/// \detail \cpp
	/// auto tbl = toml::parse("a = 3"sv);
	/// std::cout << tbl["a"] << "\n";
	/// \ecpp
	///
	/// \out
	/// 3
	/// \eout
	///
	/// \param 	doc				The TOML document to parse. Must be valid UTF-8.
	/// \param 	source_path		The path used to initialize each node's `source().path`.
	/// 						If you don't have a path (or you have no intention of using paths in diagnostics)
	/// 						then this parameter can safely be left blank.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse(std::string_view doc, std::string_view source_path = {});

	/// \brief	Parses a TOML document from a string view.
	///
	/// \detail \cpp
	/// auto tbl = toml::parse("a = 3"sv, "foo.toml");
	/// std::cout << tbl["a"] << "\n";
	/// \ecpp
	///
	/// \out
	/// 3
	/// \eout
	///
	/// \param 	doc				The TOML document to parse. Must be valid UTF-8.
	/// \param 	source_path		The path used to initialize each node's `source().path`.
	/// 						If you don't have a path (or you have no intention of using paths in diagnostics)
	/// 						then this parameter can safely be left blank.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse(std::string_view doc, std::string && source_path);

	/// \brief	Parses a TOML document from a file.
	///
	/// \detail \cpp
	/// toml::parse_result get_foo_toml()
	/// {
	///		return toml::parse_file("foo.toml");
	/// }
	/// \ecpp
	///
	/// \param 	file_path		The TOML document to parse. Must be valid UTF-8.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse_file(std::string_view file_path);

#if TOML_HAS_CHAR8

	/// \brief	Parses a TOML document from a char8_t string view.
	///
	/// \detail \cpp
	/// auto tbl = toml::parse(u8"a = 3"sv);
	/// std::cout << tbl["a"] << "\n";
	/// \ecpp
	///
	/// \out
	/// 3
	/// \eout
	///
	/// \param 	doc				The TOML document to parse. Must be valid UTF-8.
	/// \param 	source_path		The path used to initialize each node's `source().path`.
	/// 						If you don't have a path (or you have no intention of using paths in diagnostics)
	/// 						then this parameter can safely be left blank.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse(std::u8string_view doc, std::string_view source_path = {});

	/// \brief	Parses a TOML document from a char8_t string view.
	///
	/// \detail \cpp
	/// auto tbl = toml::parse(u8"a = 3"sv, "foo.toml");
	/// std::cout << tbl["a"] << "\n";
	/// \ecpp
	///
	/// \out
	/// 3
	/// \eout
	///
	/// \param 	doc				The TOML document to parse. Must be valid UTF-8.
	/// \param 	source_path		The path used to initialize each node's `source().path`.
	/// 						If you don't have a path (or you have no intention of using paths in diagnostics)
	/// 						then this parameter can safely be left blank.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse(std::u8string_view doc, std::string && source_path);

	/// \brief	Parses a TOML document from a file.
	///
	/// \detail \cpp
	/// toml::parse_result get_foo_toml()
	/// {
	///		return toml::parse_file(u8"foo.toml");
	/// }
	/// \ecpp
	///
	/// \param 	file_path		The TOML document to parse. Must be valid UTF-8.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse_file(std::u8string_view file_path);

#endif // TOML_HAS_CHAR8

#if TOML_ENABLE_WINDOWS_COMPAT

	/// \brief	Parses a TOML document from a string view.
	///
	/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
	///
	/// \detail \cpp
	/// auto tbl = toml::parse("a = 3"sv, L"foo.toml");
	/// std::cout << tbl["a"] << "\n";
	/// \ecpp
	///
	/// \out
	/// 3
	/// \eout
	///
	/// \param 	doc				The TOML document to parse. Must be valid UTF-8.
	/// \param 	source_path		The path used to initialize each node's `source().path`.
	/// 						If you don't have a path (or you have no intention of using paths in diagnostics)
	/// 						then this parameter can safely be left blank.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse(std::string_view doc, std::wstring_view source_path);

	/// \brief	Parses a TOML document from a stream.
	///
	/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
	///
	/// \detail \cpp
	/// std::stringstream ss;
	/// ss << "a = 3"sv;
	///
	/// auto tbl = toml::parse(ss);
	/// std::cout << tbl["a"] << "\n";
	/// \ecpp
	///
	/// \out
	/// 3
	/// \eout
	///
	/// \param 	doc				The TOML document to parse. Must be valid UTF-8.
	/// \param 	source_path		The path used to initialize each node's `source().path`.
	/// 						If you don't have a path (or you have no intention of using paths in diagnostics)
	/// 						then this parameter can safely be left blank.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse(std::istream & doc, std::wstring_view source_path);

	/// \brief	Parses a TOML document from a file.
	///
	/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
	///
	/// \detail \cpp
	/// toml::parse_result get_foo_toml()
	/// {
	///		return toml::parse_file(L"foo.toml");
	/// }
	/// \ecpp
	///
	/// \param 	file_path		The TOML document to parse. Must be valid UTF-8.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse_file(std::wstring_view file_path);

#endif // TOML_ENABLE_WINDOWS_COMPAT

#if TOML_HAS_CHAR8 && TOML_ENABLE_WINDOWS_COMPAT

	/// \brief	Parses a TOML document from a char8_t string view.
	///
	/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
	///
	/// \detail \cpp
	/// auto tbl = toml::parse(u8"a = 3"sv, L"foo.toml");
	/// std::cout << tbl["a"] << "\n";
	/// \ecpp
	///
	/// \out
	/// 3
	/// \eout
	///
	/// \param 	doc				The TOML document to parse. Must be valid UTF-8.
	/// \param 	source_path		The path used to initialize each node's `source().path`.
	/// 						If you don't have a path (or you have no intention of using paths in diagnostics)
	/// 						then this parameter can safely be left blank.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse(std::u8string_view doc, std::wstring_view source_path);

#endif // TOML_HAS_CHAR8 && TOML_ENABLE_WINDOWS_COMPAT

	/// \brief	Parses a TOML document from a stream.
	///
	/// \detail \cpp
	/// std::stringstream ss;
	/// ss << "a = 3"sv;
	///
	/// auto tbl = toml::parse(ss);
	/// std::cout << tbl["a"] << "\n";
	/// \ecpp
	///
	/// \out
	/// 3
	/// \eout
	///
	/// \param 	doc				The TOML document to parse. Must be valid UTF-8.
	/// \param 	source_path		The path used to initialize each node's `source().path`.
	/// 						If you don't have a path (or you have no intention of using paths in diagnostics)
	/// 						then this parameter can safely be left blank.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse(std::istream & doc, std::string_view source_path = {});

	/// \brief	Parses a TOML document from a stream.
	///
	/// \detail \cpp
	/// std::stringstream ss;
	/// ss << "a = 3"sv;
	///
	/// auto tbl = toml::parse(ss, "foo.toml");
	/// std::cout << tbl["a"] << "\n";
	/// \ecpp
	///
	/// \out
	/// 3
	/// \eout
	///
	/// \param 	doc				The TOML document to parse. Must be valid UTF-8.
	/// \param 	source_path		The path used to initialize each node's `source().path`.
	/// 						If you don't have a path (or you have no intention of using paths in diagnostics)
	/// 						then this parameter can safely be left blank.
	///
	/// \returns	\conditional_return{With exceptions}
	///				A toml::table.
	/// 			\conditional_return{Without exceptions}
	///				A toml::parse_result.
	TOML_NODISCARD
	TOML_EXPORTED_FREE_FUNCTION
	parse_result parse(std::istream & doc, std::string && source_path);

	TOML_ABI_NAMESPACE_END; // TOML_EXCEPTIONS

	inline namespace literals
	{
		TOML_ABI_NAMESPACE_BOOL(TOML_EXCEPTIONS, lit_ex, lit_noex);

		/// \brief	Parses TOML data from a string literal.
		///
		/// \detail \cpp
		/// using namespace toml::literals;
		///
		/// auto tbl = "a = 3"_toml;
		/// std::cout << tbl["a"] << "\n";
		/// \ecpp
		///
		/// \out
		/// 3
		/// \eout
		///
		/// \param 	str	The string data. Must be valid UTF-8.
		/// \param 	len	The string length.
		///
		/// \returns	\conditional_return{With exceptions}
		///				A toml::table.
		/// 			\conditional_return{Without exceptions}
		///				A toml::parse_result.
		TOML_NODISCARD
		inline parse_result operator"" _toml(const char* str, size_t len)
		{
			return parse(std::string_view{ str, len });
		}

#if TOML_HAS_CHAR8

		/// \brief	Parses TOML data from a UTF-8 string literal.
		///
		/// \detail \cpp
		/// using namespace toml::literals;
		///
		/// auto tbl = u8"a = 3"_toml;
		/// std::cout << tbl["a"] << "\n";
		/// \ecpp
		///
		/// \out
		/// 3
		/// \eout
		///
		/// \param 	str	The string data. Must be valid UTF-8.
		/// \param 	len	The string length.
		///
		/// \returns	\conditional_return{With exceptions}
		///				A toml::table.
		/// 			\conditional_return{Without exceptions}
		///				A toml::parse_result.
		TOML_NODISCARD
		inline parse_result operator"" _toml(const char8_t* str, size_t len)
		{
			return parse(std::u8string_view{ str, len });
		}

#endif // TOML_HAS_CHAR8

		TOML_ABI_NAMESPACE_END; // TOML_EXCEPTIONS
	}
}
TOML_NAMESPACE_END;

#include "header_end.h"
#endif // TOML_ENABLE_PARSER
