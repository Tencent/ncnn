//# This file is a part of toml++ and is subject to the the terms of the MIT license.
//# Copyright (c) Mark Gillard <mark.gillard@outlook.com.au>
//# See https://github.com/marzer/tomlplusplus/blob/master/LICENSE for the full license text.
// SPDX-License-Identifier: MIT
#pragma once

#include "preprocessor.h"
#if TOML_ENABLE_FORMATTERS

#include "std_vector.h"
#include "formatter.h"
#include "header_start.h"

TOML_NAMESPACE_START
{
	/// \brief	A wrapper for printing TOML objects out to a stream as formatted TOML.
	///
	/// \availability This class is only available when #TOML_ENABLE_FORMATTERS is enabled.
	///
	/// \remarks You generally don't need to create an instance of this class explicitly; the stream
	/// 		 operators of the TOML node types already print themselves out using this formatter.
	///
	/// \detail \cpp
	/// auto tbl = toml::table{
	///		{ "description", "This is some TOML, yo." },
	///		{ "fruit", toml::array{ "apple", "orange", "pear" } },
	///		{ "numbers", toml::array{ 1, 2, 3, 4, 5 } },
	///		{ "table", toml::table{ { "foo", "bar" } } }
	/// };
	///
	/// // these two lines are equivalent:
	///	std::cout << toml::toml_formatter{ tbl } << "\n";
	///	std::cout << tbl << "\n";
	/// \ecpp
	///
	/// \out
	/// description = "This is some TOML, yo."
	/// fruit = ["apple", "orange", "pear"]
	/// numbers = [1, 2, 3, 4, 5]
	///
	/// [table]
	/// foo = "bar"
	/// \eout
	class TOML_EXPORTED_CLASS toml_formatter : impl::formatter
	{
	  private:
		/// \cond

		using base = impl::formatter;
		std::vector<const key*> key_path_;
		bool pending_table_separator_ = false;

		TOML_EXPORTED_MEMBER_FUNCTION
		void print_pending_table_separator();

		TOML_EXPORTED_MEMBER_FUNCTION
		void print(const key&);

		TOML_EXPORTED_MEMBER_FUNCTION
		void print_inline(const toml::table&);

		TOML_EXPORTED_MEMBER_FUNCTION
		void print(const toml::array&);

		TOML_EXPORTED_MEMBER_FUNCTION
		void print(const toml::table&);

		TOML_EXPORTED_MEMBER_FUNCTION
		void print();

		static constexpr impl::formatter_constants constants = { format_flags::none, // mandatory
																 format_flags::none, // ignored
																 "inf"sv,
																 "-inf"sv,
																 "nan"sv,
																 "true"sv,
																 "false"sv };

		/// \endcond

	  public:
		/// \brief	The default flags for a toml_formatter.
		static constexpr format_flags default_flags = constants.mandatory_flags				   //
													| format_flags::allow_literal_strings	   //
													| format_flags::allow_multi_line_strings   //
													| format_flags::allow_unicode_strings	   //
													| format_flags::allow_real_tabs_in_strings //
													| format_flags::allow_binary_integers	   //
													| format_flags::allow_octal_integers	   //
													| format_flags::allow_hexadecimal_integers //
													| format_flags::indentation;

		/// \brief	Constructs a TOML formatter and binds it to a TOML object.
		///
		/// \param 	source	The source TOML object.
		/// \param 	flags 	Format option flags.
		TOML_NODISCARD_CTOR
		explicit toml_formatter(const toml::node& source, format_flags flags = default_flags) noexcept
			: base{ &source, nullptr, constants, { flags, "    "sv } }
		{}

#if defined(DOXYGEN) || (TOML_ENABLE_PARSER && !TOML_EXCEPTIONS)

		/// \brief	Constructs a TOML formatter and binds it to a toml::parse_result.
		///
		/// \availability This constructor is only available when exceptions are disabled.
		///
		/// \attention Formatting a failed parse result will simply dump the error message out as-is.
		///		This will not be valid TOML, but at least gives you something to log or show up in diagnostics:
		/// \cpp
		/// std::cout << toml::toml_formatter{ toml::parse("a = 'b'"sv) } // ok
		///           << "\n\n"
		///           << toml::toml_formatter{ toml::parse("a = "sv) } // malformed
		///           << "\n";
		/// \ecpp
		/// \out
		/// a = 'b'
		///
		/// Error while parsing key-value pair: encountered end-of-file
		///         (error occurred at line 1, column 5)
		/// \eout
		/// Use the library with exceptions if you want to avoid this scenario.
		///
		/// \param 	result	The parse result.
		/// \param 	flags 	Format option flags.
		TOML_NODISCARD_CTOR
		explicit toml_formatter(const toml::parse_result& result, format_flags flags = default_flags) noexcept
			: base{ nullptr, &result, constants, { flags, "    "sv } }
		{}

#endif

		/// \brief	Prints the bound TOML object out to the stream as formatted TOML.
		friend std::ostream& operator<<(std::ostream& lhs, toml_formatter& rhs)
		{
			rhs.attach(lhs);
			rhs.key_path_.clear();
			rhs.print();
			rhs.detach();
			return lhs;
		}

		/// \brief	Prints the bound TOML object out to the stream as formatted TOML (rvalue overload).
		friend std::ostream& operator<<(std::ostream& lhs, toml_formatter&& rhs)
		{
			return lhs << rhs; // as lvalue
		}
	};
}
TOML_NAMESPACE_END;

#include "header_end.h"
#endif // TOML_ENABLE_FORMATTERS
