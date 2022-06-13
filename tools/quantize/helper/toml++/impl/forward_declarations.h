//# This file is a part of toml++ and is subject to the the terms of the MIT license.
//# Copyright (c) Mark Gillard <mark.gillard@outlook.com.au>
//# See https://github.com/marzer/tomlplusplus/blob/master/LICENSE for the full license text.
// SPDX-License-Identifier: MIT
#pragma once

#include "std_string.h"
#include "std_new.h"
TOML_DISABLE_WARNINGS;
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cfloat>
#include <climits>
#include <cmath>
#include <limits>
#include <memory>
#include <iosfwd>
#include <type_traits>
TOML_ENABLE_WARNINGS;
#include "header_start.h"

//#---------------------------------------------------------------------------------------------------------------------
//# ENVIRONMENT GROUND-TRUTHS
//#---------------------------------------------------------------------------------------------------------------------
/// \cond

#ifndef TOML_DISABLE_ENVIRONMENT_CHECKS
#define TOML_ENV_MESSAGE                                                                                               \
	"If you're seeing this error it's because you're building toml++ for an environment that doesn't conform to "      \
	"one of the 'ground truths' assumed by the library. Essentially this just means that I don't have the "            \
	"resources to test on more platforms, but I wish I did! You can try disabling the checks by defining "             \
	"TOML_DISABLE_ENVIRONMENT_CHECKS, but your mileage may vary. Please consider filing an issue at "                  \
	"https://github.com/marzer/tomlplusplus/issues to help me improve support for your target environment. "           \
	"Thanks!"

static_assert(CHAR_BIT == 8, TOML_ENV_MESSAGE);
static_assert(FLT_RADIX == 2, TOML_ENV_MESSAGE);
static_assert('A' == 65, TOML_ENV_MESSAGE);
static_assert(sizeof(double) == 8, TOML_ENV_MESSAGE);
static_assert(std::numeric_limits<double>::is_iec559, TOML_ENV_MESSAGE);
static_assert(std::numeric_limits<double>::digits == 53, TOML_ENV_MESSAGE);
static_assert(std::numeric_limits<double>::digits10 == 15, TOML_ENV_MESSAGE);

#undef TOML_ENV_MESSAGE
#endif // !TOML_DISABLE_ENVIRONMENT_CHECKS

/// \endcond

//#---------------------------------------------------------------------------------------------------------------------
//# UNDOCUMENTED TYPEDEFS AND FORWARD DECLARATIONS
//#---------------------------------------------------------------------------------------------------------------------
/// \cond
// undocumented forward declarations are hidden from doxygen because they fuck it up =/

namespace toml // non-abi namespace; this is not an error
{
	using ::std::size_t;
	using ::std::intptr_t;
	using ::std::uintptr_t;
	using ::std::ptrdiff_t;
	using ::std::nullptr_t;
	using ::std::int8_t;
	using ::std::int16_t;
	using ::std::int32_t;
	using ::std::int64_t;
	using ::std::uint8_t;
	using ::std::uint16_t;
	using ::std::uint32_t;
	using ::std::uint64_t;
	using ::std::uint_least32_t;
	using ::std::uint_least64_t;
}

TOML_NAMESPACE_START
{
	struct date;
	struct time;
	struct time_offset;

	TOML_ABI_NAMESPACE_BOOL(TOML_HAS_CUSTOM_OPTIONAL_TYPE, custopt, stdopt);
	struct date_time;
	TOML_ABI_NAMESPACE_END;

	struct source_position;
	struct source_region;

	class node;
	template <typename>
	class node_view;

	class key;
	class array;
	class table;
	template <typename>
	class value;

	class toml_formatter;
	class json_formatter;
	class yaml_formatter;

	TOML_ABI_NAMESPACE_BOOL(TOML_EXCEPTIONS, ex, noex);
#if TOML_EXCEPTIONS
	using parse_result = table;
#else
	class parse_result;
#endif
	TOML_ABI_NAMESPACE_END; // TOML_EXCEPTIONS
}
TOML_NAMESPACE_END;

TOML_IMPL_NAMESPACE_START
{
	using node_ptr = std::unique_ptr<node>;

	TOML_ABI_NAMESPACE_BOOL(TOML_EXCEPTIONS, impl_ex, impl_noex);
	class parser;
	TOML_ABI_NAMESPACE_END; // TOML_EXCEPTIONS

	// clang-format off

	inline constexpr std::string_view control_char_escapes[] =
	{
		"\\u0000"sv,
		"\\u0001"sv,
		"\\u0002"sv,
		"\\u0003"sv,
		"\\u0004"sv,
		"\\u0005"sv,
		"\\u0006"sv,
		"\\u0007"sv,
		"\\b"sv,
		"\\t"sv,
		"\\n"sv,
		"\\u000B"sv,
		"\\f"sv,
		"\\r"sv,
		"\\u000E"sv,
		"\\u000F"sv,
		"\\u0010"sv,
		"\\u0011"sv,
		"\\u0012"sv,
		"\\u0013"sv,
		"\\u0014"sv,
		"\\u0015"sv,
		"\\u0016"sv,
		"\\u0017"sv,
		"\\u0018"sv,
		"\\u0019"sv,
		"\\u001A"sv,
		"\\u001B"sv,
		"\\u001C"sv,
		"\\u001D"sv,
		"\\u001E"sv,
		"\\u001F"sv,
	};

	inline constexpr std::string_view node_type_friendly_names[] =
	{
		"none"sv,
		"table"sv,
		"array"sv,
		"string"sv,
		"integer"sv,
		"floating-point"sv,
		"boolean"sv,
		"date"sv,
		"time"sv,
		"date-time"sv
	};

	// clang-format on
}
TOML_IMPL_NAMESPACE_END;

#if TOML_ABI_NAMESPACES
#if TOML_EXCEPTIONS
#define TOML_PARSER_TYPENAME TOML_NAMESPACE::impl::impl_ex::parser
#else
#define TOML_PARSER_TYPENAME TOML_NAMESPACE::impl::impl_noex::parser
#endif
#else
#define TOML_PARSER_TYPENAME TOML_NAMESPACE::impl::parser
#endif

/// \endcond

//#---------------------------------------------------------------------------------------------------------------------
//# DOCUMENTED TYPEDEFS AND FORWARD DECLARATIONS
//#---------------------------------------------------------------------------------------------------------------------

/// \brief	The root namespace for all toml++ functions and types.
namespace toml
{
}

TOML_NAMESPACE_START // abi namespace
{
	/// \brief	Convenience literal operators for working with toml++.
	///
	/// \detail This namespace exists so you can safely hoist the toml++ literal operators into another scope
	/// 		 without dragging in everything from the toml namespace: \cpp
	///
	/// #include <toml++/toml.h>
	///	using namespace toml::literals;
	///
	///	int main()
	///	{
	///		toml::table tbl = "vals = [1, 2, 3]"_toml;
	///
	///		// ... do stuff with the table generated by the "_toml" literal ...
	///
	///		return 0;
	///	}
	/// \ecpp
	///
	inline namespace literals
	{
	}

	/// \brief	TOML node type identifiers.
	enum class TOML_CLOSED_ENUM node_type : uint8_t
	{
		none,			///< Not-a-node.
		table,			///< The node is a toml::table.
		array,			///< The node is a toml::array.
		string,			///< The node is a toml::value<std::string>.
		integer,		///< The node is a toml::value<int64_t>.
		floating_point, ///< The node is a toml::value<double>.
		boolean,		///< The node is a toml::value<bool>.
		date,			///< The node is a toml::value<date>.
		time,			///< The node is a toml::value<time>.
		date_time		///< The node is a toml::value<date_time>.
	};

	/// \brief	Pretty-prints the value of a node_type to a stream.
	///
	/// \detail \cpp
	/// auto arr = toml::array{ 1, 2.0, "3", false };
	/// for (size_t i = 0; i < arr.size() i++)
	/// 	std::cout << "Element ["sv << i << "] is: "sv << arr[i].type() << "\n";
	/// \ecpp
	///
	/// \out
	/// Element [0] is: integer
	/// Element [1] is: floating-point
	/// Element [2] is: string
	/// Element [3] is: boolean
	/// \eout
	template <typename Char>
	inline std::basic_ostream<Char>& operator<<(std::basic_ostream<Char>& lhs, node_type rhs)
	{
		using underlying_t = std::underlying_type_t<node_type>;
		const auto str	   = impl::node_type_friendly_names[static_cast<underlying_t>(rhs)];
		if constexpr (std::is_same_v<Char, char>)
			return lhs << str;
		else
		{
			if constexpr (sizeof(Char) == 1)
				return lhs << std::basic_string_view<Char>{ reinterpret_cast<const Char*>(str.data()), str.length() };
			else
				return lhs << str.data();
		}
	}

	/// \brief Metadata associated with TOML values.
	enum class TOML_OPEN_FLAGS_ENUM value_flags : uint16_t // being an "OPEN" flags enum is not an error
	{
		/// \brief None.
		none,

		/// \brief Format integer values as binary.
		format_as_binary = 1,

		/// \brief Format integer values as octal.
		format_as_octal = 2,

		/// \brief Format integer values as hexadecimal.
		format_as_hexadecimal = 3,
	};
	TOML_MAKE_FLAGS(value_flags);

	/// \brief Special #toml::value_flags constant used for array + table insert functions to specify that any value
	/// nodes being copied should not have their flags property overridden by the inserting function's `flags` argument.
	inline constexpr value_flags preserve_source_value_flags =
		POXY_IMPLEMENTATION_DETAIL(value_flags{ static_cast<std::underlying_type_t<value_flags>>(-1) });

	/// \brief	Format flags for modifying how TOML data is printed to streams.
	///
	/// \note	Formatters may disregard/override any of these flags according to the requirements of their
	///			output target (e.g. #toml::json_formatter will always apply quotes to dates and times).
	enum class TOML_CLOSED_FLAGS_ENUM format_flags : uint64_t
	{
		/// \brief None.
		none,

		/// \brief Dates and times will be emitted as quoted strings.
		quote_dates_and_times = (1ull << 0),

		/// \brief Infinities and NaNs will be emitted as quoted strings.
		quote_infinities_and_nans = (1ull << 1),

		/// \brief Strings will be emitted as single-quoted literal strings where possible.
		allow_literal_strings = (1ull << 2),

		/// \brief Strings containing newlines will be emitted as triple-quoted 'multi-line' strings where possible.
		allow_multi_line_strings = (1ull << 3),

		/// \brief Allow real tab characters in string literals (as opposed to the escaped form `\t`).
		allow_real_tabs_in_strings = (1ull << 4),

		/// \brief Allow non-ASCII characters in strings (as opposed to their escaped form, e.g. `\u00DA`).
		allow_unicode_strings = (1ull << 5),

		/// \brief Allow integers with #value_flags::format_as_binary to be emitted as binary.
		allow_binary_integers = (1ull << 6),

		/// \brief Allow integers with #value_flags::format_as_octal to be emitted as octal.
		allow_octal_integers = (1ull << 7),

		/// \brief Allow integers with #value_flags::format_as_hexadecimal to be emitted as hexadecimal.
		allow_hexadecimal_integers = (1ull << 8),

		/// \brief Apply indentation to tables nested within other tables/arrays.
		indent_sub_tables = (1ull << 9),

		/// \brief Apply indentation to array elements when the array is forced to wrap over multiple lines.
		indent_array_elements = (1ull << 10),

		/// \brief Combination mask of all indentation-enabling flags.
		indentation = indent_sub_tables | indent_array_elements,

		/// \brief Emit floating-point values with relaxed (human-friendly) precision.
		/// \warning	Setting this flag may cause serialized documents to no longer round-trip correctly
		///				since floats might have a less precise value upon being written out than they did when being
		///				read in. Use this flag at your own risk.
		relaxed_float_precision = (1ull << 11),
	};
	TOML_MAKE_FLAGS(format_flags);

	/// \brief	Helper class for suppressing move-construction in single-argument array constructors.
	///
	/// \detail \cpp
	/// // desired result: [ [ 42 ] ]
	/// auto bad = toml::array{ toml::array{ 42 } }
	/// auto good = toml::array{ toml::inserter{ toml::array{ 42 } } }
	/// std::cout << "bad: " << bad << "\n";
	/// std::cout << "good:" << good << "\n";
	/// \ecpp
	/// \out
	/// bad:  [ 42 ]
	/// good: [ [ 42 ] ]
	/// \eout
	///
	/// \see toml::array
	template <typename T>
	struct TOML_TRIVIAL_ABI inserter
	{
		static_assert(std::is_reference_v<T>);

		T value;
	};
	template <typename T>
	inserter(T &&) -> inserter<T&&>;
	template <typename T>
	inserter(T&) -> inserter<T&>;

	/// \brief The 'default' formatter used by TOML objects when they are printed to a stream.
	/// \detail This is an alias for #toml::toml_formatter.
	using default_formatter = toml_formatter;
}
TOML_NAMESPACE_END;

//#---------------------------------------------------------------------------------------------------------------------
//# METAFUNCTIONS & TYPE TRAITS
//#---------------------------------------------------------------------------------------------------------------------
/// \cond
TOML_IMPL_NAMESPACE_START
{
	template <typename T>
	using remove_cvref = std::remove_cv_t<std::remove_reference_t<T>>;

	template <typename... T>
	using common_signed_type = std::common_type_t<std::make_signed_t<T>...>;

	template <typename T, typename... U>
	inline constexpr bool is_one_of = (false || ... || std::is_same_v<T, U>);

	template <typename... T>
	inline constexpr bool all_integral = (std::is_integral_v<T> && ...);

	template <typename T>
	inline constexpr bool is_cvref = std::is_reference_v<T> || std::is_const_v<T> || std::is_volatile_v<T>;

	template <typename T>
	inline constexpr bool is_wide_string =
		is_one_of<std::decay_t<T>, const wchar_t*, wchar_t*, std::wstring_view, std::wstring>;

	template <typename T>
	inline constexpr bool value_retrieval_is_nothrow = !std::is_same_v<remove_cvref<T>, std::string>
#if TOML_HAS_CHAR8
													&& !std::is_same_v<remove_cvref<T>, std::u8string>
#endif

													&& !is_wide_string<T>;

	template <typename, typename>
	struct copy_ref_;
	template <typename Dest, typename Src>
	using copy_ref = typename copy_ref_<Dest, Src>::type;

	template <typename Dest, typename Src>
	struct copy_ref_
	{
		using type = Dest;
	};

	template <typename Dest, typename Src>
	struct copy_ref_<Dest, Src&>
	{
		using type = std::add_lvalue_reference_t<Dest>;
	};

	template <typename Dest, typename Src>
	struct copy_ref_<Dest, Src&&>
	{
		using type = std::add_rvalue_reference_t<Dest>;
	};

	template <typename, typename>
	struct copy_cv_;
	template <typename Dest, typename Src>
	using copy_cv = typename copy_cv_<Dest, Src>::type;

	template <typename Dest, typename Src>
	struct copy_cv_
	{
		using type = Dest;
	};

	template <typename Dest, typename Src>
	struct copy_cv_<Dest, const Src>
	{
		using type = std::add_const_t<Dest>;
	};

	template <typename Dest, typename Src>
	struct copy_cv_<Dest, volatile Src>
	{
		using type = std::add_volatile_t<Dest>;
	};

	template <typename Dest, typename Src>
	struct copy_cv_<Dest, const volatile Src>
	{
		using type = std::add_cv_t<Dest>;
	};

	template <typename Dest, typename Src>
	using copy_cvref =
		copy_ref<copy_ref<copy_cv<std::remove_reference_t<Dest>, std::remove_reference_t<Src>>, Dest>, Src>;

	template <typename T>
	inline constexpr bool dependent_false = false;

	template <typename T, typename... U>
	inline constexpr bool first_is_same = false;
	template <typename T, typename... U>
	inline constexpr bool first_is_same<T, T, U...> = true;

	// general value traits
	// (as they relate to their equivalent native TOML type)
	template <typename T>
	struct value_traits
	{
		using native_type										  = void;
		static constexpr bool is_native							  = false;
		static constexpr bool is_losslessly_convertible_to_native = false;
		static constexpr bool can_represent_native				  = false;
		static constexpr bool can_partially_represent_native	  = false;
		static constexpr auto type								  = node_type::none;
	};

	template <typename T>
	struct value_traits<const T> : value_traits<T>
	{};
	template <typename T>
	struct value_traits<volatile T> : value_traits<T>
	{};
	template <typename T>
	struct value_traits<const volatile T> : value_traits<T>
	{};
	template <typename T>
	struct value_traits<T&> : value_traits<T>
	{};
	template <typename T>
	struct value_traits<T&&> : value_traits<T>
	{};

	// integer value_traits specializations - standard types
	template <typename T>
	struct integer_value_limits
	{
		static constexpr auto min = (std::numeric_limits<T>::min)();
		static constexpr auto max = (std::numeric_limits<T>::max)();
	};
	template <typename T>
	struct integer_value_traits_base : integer_value_limits<T>
	{
		using native_type				= int64_t;
		static constexpr bool is_native = std::is_same_v<T, native_type>;
		static constexpr bool is_signed = static_cast<T>(-1) < T{}; // for impls not specializing std::is_signed<T>
		static constexpr auto type		= node_type::integer;
		static constexpr bool can_partially_represent_native = true;
	};
	template <typename T>
	struct unsigned_integer_value_traits : integer_value_traits_base<T>
	{
		static constexpr bool is_losslessly_convertible_to_native =
			integer_value_limits<T>::max <= 9223372036854775807ULL;
		static constexpr bool can_represent_native = false;
	};
	template <typename T>
	struct signed_integer_value_traits : integer_value_traits_base<T>
	{
		using native_type = int64_t;
		static constexpr bool is_losslessly_convertible_to_native =
			integer_value_limits<T>::min >= (-9223372036854775807LL - 1LL)
			&& integer_value_limits<T>::max <= 9223372036854775807LL;
		static constexpr bool can_represent_native = integer_value_limits<T>::min <= (-9223372036854775807LL - 1LL)
												  && integer_value_limits<T>::max >= 9223372036854775807LL;
	};
	template <typename T, bool S = integer_value_traits_base<T>::is_signed>
	struct integer_value_traits : signed_integer_value_traits<T>
	{};
	template <typename T>
	struct integer_value_traits<T, false> : unsigned_integer_value_traits<T>
	{};
	template <>
	struct value_traits<signed char> : integer_value_traits<signed char>
	{};
	template <>
	struct value_traits<unsigned char> : integer_value_traits<unsigned char>
	{};
	template <>
	struct value_traits<signed short> : integer_value_traits<signed short>
	{};
	template <>
	struct value_traits<unsigned short> : integer_value_traits<unsigned short>
	{};
	template <>
	struct value_traits<signed int> : integer_value_traits<signed int>
	{};
	template <>
	struct value_traits<unsigned int> : integer_value_traits<unsigned int>
	{};
	template <>
	struct value_traits<signed long> : integer_value_traits<signed long>
	{};
	template <>
	struct value_traits<unsigned long> : integer_value_traits<unsigned long>
	{};
	template <>
	struct value_traits<signed long long> : integer_value_traits<signed long long>
	{};
	template <>
	struct value_traits<unsigned long long> : integer_value_traits<unsigned long long>
	{};
	static_assert(value_traits<int64_t>::is_native);
	static_assert(value_traits<int64_t>::is_signed);
	static_assert(value_traits<int64_t>::is_losslessly_convertible_to_native);
	static_assert(value_traits<int64_t>::can_represent_native);
	static_assert(value_traits<int64_t>::can_partially_represent_native);

	// integer value_traits specializations - non-standard types
#ifdef TOML_INT128
	template <>
	struct integer_value_limits<TOML_INT128>
	{
		static constexpr TOML_INT128 max =
			static_cast<TOML_INT128>((TOML_UINT128{ 1u } << ((__SIZEOF_INT128__ * CHAR_BIT) - 1)) - 1);
		static constexpr TOML_INT128 min = -max - TOML_INT128{ 1 };
	};
	template <>
	struct integer_value_limits<TOML_UINT128>
	{
		static constexpr TOML_UINT128 min = TOML_UINT128{};
		static constexpr TOML_UINT128 max =
			(2u * static_cast<TOML_UINT128>(integer_value_limits<TOML_INT128>::max)) + 1u;
	};
	template <>
	struct value_traits<TOML_INT128> : integer_value_traits<TOML_INT128>
	{};
	template <>
	struct value_traits<TOML_UINT128> : integer_value_traits<TOML_UINT128>
	{};
#endif
#ifdef TOML_SMALL_INT_TYPE
	template <>
	struct value_traits<TOML_SMALL_INT_TYPE> : signed_integer_value_traits<TOML_SMALL_INT_TYPE>
	{};
#endif

	// floating-point value_traits specializations - standard types
	template <typename T>
	struct float_value_limits
	{
		static constexpr bool is_iec559 = std::numeric_limits<T>::is_iec559;
		static constexpr int digits		= std::numeric_limits<T>::digits;
		static constexpr int digits10	= std::numeric_limits<T>::digits10;
	};
	template <typename T>
	struct float_value_traits : float_value_limits<T>
	{
		using native_type				= double;
		static constexpr bool is_native = std::is_same_v<T, native_type>;
		static constexpr bool is_signed = true;

		static constexpr bool is_losslessly_convertible_to_native = float_value_limits<T>::is_iec559
																 && float_value_limits<T>::digits <= 53
																 && float_value_limits<T>::digits10 <= 15;

		static constexpr bool can_represent_native = float_value_limits<T>::is_iec559
												  && float_value_limits<T>::digits >= 53	// DBL_MANT_DIG
												  && float_value_limits<T>::digits10 >= 15; // DBL_DIG

		static constexpr bool can_partially_represent_native // 32-bit float values
			= float_value_limits<T>::is_iec559				 //
		   && float_value_limits<T>::digits >= 24			 //
		   && float_value_limits<T>::digits10 >= 6;

		static constexpr auto type = node_type::floating_point;
	};
	template <>
	struct value_traits<float> : float_value_traits<float>
	{};
	template <>
	struct value_traits<double> : float_value_traits<double>
	{};
	template <>
	struct value_traits<long double> : float_value_traits<long double>
	{};
	template <int mant_dig, int dig>
	struct extended_float_value_limits
	{
		static constexpr bool is_iec559 = true;
		static constexpr int digits		= mant_dig;
		static constexpr int digits10	= dig;
	};
	static_assert(value_traits<double>::is_native);
	static_assert(value_traits<double>::is_losslessly_convertible_to_native);
	static_assert(value_traits<double>::can_represent_native);
	static_assert(value_traits<double>::can_partially_represent_native);

	// floating-point value_traits specializations - non-standard types
#ifdef TOML_FP16
	template <>
	struct float_value_limits<TOML_FP16> : extended_float_value_limits<__FLT16_MANT_DIG__, __FLT16_DIG__>
	{};
	template <>
	struct value_traits<TOML_FP16> : float_value_traits<TOML_FP16>
	{};
#endif
#ifdef TOML_FLOAT16
	template <>
	struct float_value_limits<TOML_FLOAT16> : extended_float_value_limits<__FLT16_MANT_DIG__, __FLT16_DIG__>
	{};
	template <>
	struct value_traits<TOML_FLOAT16> : float_value_traits<TOML_FLOAT16>
	{};
#endif
#ifdef TOML_FLOAT128
	template <>
	struct float_value_limits<TOML_FLOAT128> : extended_float_value_limits<__FLT128_MANT_DIG__, __FLT128_DIG__>
	{};
	template <>
	struct value_traits<TOML_FLOAT128> : float_value_traits<TOML_FLOAT128>
	{};
#endif
#ifdef TOML_SMALL_FLOAT_TYPE
	template <>
	struct value_traits<TOML_SMALL_FLOAT_TYPE> : float_value_traits<TOML_SMALL_FLOAT_TYPE>
	{};
#endif

	// string value_traits specializations - char-based strings
	template <typename T>
	struct string_value_traits
	{
		using native_type										  = std::string;
		static constexpr bool is_native							  = std::is_same_v<T, native_type>;
		static constexpr bool is_losslessly_convertible_to_native = true;
		static constexpr bool can_represent_native =
			!std::is_array_v<T> && (!std::is_pointer_v<T> || std::is_const_v<std::remove_pointer_t<T>>);
		static constexpr bool can_partially_represent_native = can_represent_native;
		static constexpr auto type							 = node_type::string;
	};
	template <>
	struct value_traits<std::string> : string_value_traits<std::string>
	{};
	template <>
	struct value_traits<std::string_view> : string_value_traits<std::string_view>
	{};
	template <>
	struct value_traits<const char*> : string_value_traits<const char*>
	{};
	template <size_t N>
	struct value_traits<const char[N]> : string_value_traits<const char[N]>
	{};
	template <>
	struct value_traits<char*> : string_value_traits<char*>
	{};
	template <size_t N>
	struct value_traits<char[N]> : string_value_traits<char[N]>
	{};

	// string value_traits specializations - char8_t-based strings
#if TOML_HAS_CHAR8
	template <>
	struct value_traits<std::u8string> : string_value_traits<std::u8string>
	{};
	template <>
	struct value_traits<std::u8string_view> : string_value_traits<std::u8string_view>
	{};
	template <>
	struct value_traits<const char8_t*> : string_value_traits<const char8_t*>
	{};
	template <size_t N>
	struct value_traits<const char8_t[N]> : string_value_traits<const char8_t[N]>
	{};
	template <>
	struct value_traits<char8_t*> : string_value_traits<char8_t*>
	{};
	template <size_t N>
	struct value_traits<char8_t[N]> : string_value_traits<char8_t[N]>
	{};
#endif

	// string value_traits specializations - wchar_t-based strings on Windows
#if TOML_ENABLE_WINDOWS_COMPAT
	template <typename T>
	struct wstring_value_traits
	{
		using native_type										  = std::string;
		static constexpr bool is_native							  = false;
		static constexpr bool is_losslessly_convertible_to_native = true;							 // narrow
		static constexpr bool can_represent_native				  = std::is_same_v<T, std::wstring>; // widen
		static constexpr bool can_partially_represent_native	  = can_represent_native;
		static constexpr auto type								  = node_type::string;
	};
	template <>
	struct value_traits<std::wstring> : wstring_value_traits<std::wstring>
	{};
	template <>
	struct value_traits<std::wstring_view> : wstring_value_traits<std::wstring_view>
	{};
	template <>
	struct value_traits<const wchar_t*> : wstring_value_traits<const wchar_t*>
	{};
	template <size_t N>
	struct value_traits<const wchar_t[N]> : wstring_value_traits<const wchar_t[N]>
	{};
	template <>
	struct value_traits<wchar_t*> : wstring_value_traits<wchar_t*>
	{};
	template <size_t N>
	struct value_traits<wchar_t[N]> : wstring_value_traits<wchar_t[N]>
	{};
#endif

	// other 'native' value_traits specializations
	template <typename T, node_type NodeType>
	struct native_value_traits
	{
		using native_type										  = T;
		static constexpr bool is_native							  = true;
		static constexpr bool is_losslessly_convertible_to_native = true;
		static constexpr bool can_represent_native				  = true;
		static constexpr bool can_partially_represent_native	  = true;
		static constexpr auto type								  = NodeType;
	};
	template <>
	struct value_traits<bool> : native_value_traits<bool, node_type::boolean>
	{};
	template <>
	struct value_traits<date> : native_value_traits<date, node_type::date>
	{};
	template <>
	struct value_traits<time> : native_value_traits<time, node_type::time>
	{};
	template <>
	struct value_traits<date_time> : native_value_traits<date_time, node_type::date_time>
	{};

	// native value category queries
	template <typename T>
	using native_type_of = typename value_traits<T>::native_type;
	template <typename T>
	inline constexpr bool is_native = value_traits<T>::is_native;
	template <typename T>
	inline constexpr bool can_represent_native = value_traits<T>::can_represent_native;
	template <typename T>
	inline constexpr bool can_partially_represent_native = value_traits<T>::can_partially_represent_native;
	template <typename T>
	inline constexpr bool is_losslessly_convertible_to_native = value_traits<T>::is_losslessly_convertible_to_native;
	template <typename T, typename... U>
	inline constexpr bool is_natively_one_of = is_one_of<native_type_of<T>, U...>;

	// native value types => nodes
	template <typename T>
	struct node_wrapper
	{
		using type = T;
	};
	template <typename T>
	struct node_wrapper<const T>
	{
		using type = std::add_const_t<typename node_wrapper<T>::type>;
	};
	template <typename T>
	struct node_wrapper<volatile T>
	{
		using type = std::add_volatile_t<typename node_wrapper<T>::type>;
	};
	template <typename T>
	struct node_wrapper<const volatile T>
	{
		using type = std::add_const_t<std::add_volatile_t<typename node_wrapper<T>::type>>;
	};
	template <>
	struct node_wrapper<std::string>
	{
		using type = value<std::string>;
	};
	template <>
	struct node_wrapper<int64_t>
	{
		using type = value<int64_t>;
	};
	template <>
	struct node_wrapper<double>
	{
		using type = value<double>;
	};
	template <>
	struct node_wrapper<bool>
	{
		using type = value<bool>;
	};
	template <>
	struct node_wrapper<date>
	{
		using type = value<date>;
	};
	template <>
	struct node_wrapper<time>
	{
		using type = value<time>;
	};
	template <>
	struct node_wrapper<date_time>
	{
		using type = value<date_time>;
	};
	template <typename T>
	using wrap_node = typename node_wrapper<std::remove_reference_t<T>>::type;

	// nodes => native value types
	template <typename T>
	struct node_unwrapper
	{
		using type = T;
	};
	template <typename T>
	struct node_unwrapper<value<T>>
	{
		using type = T;
	};
	template <typename T>
	struct node_unwrapper<const value<T>>
	{
		using type = std::add_const_t<T>;
	};
	template <typename T>
	struct node_unwrapper<volatile value<T>>
	{
		using type = std::add_volatile_t<T>;
	};
	template <typename T>
	struct node_unwrapper<const volatile value<T>>
	{
		using type = std::add_volatile_t<std::add_const_t<T>>;
	};
	template <typename T>
	using unwrap_node = typename node_unwrapper<std::remove_reference_t<T>>::type;

	template <typename T>
	struct node_type_getter
	{
		static constexpr auto value = value_traits<T>::type;
	};
	template <>
	struct node_type_getter<table>
	{
		static constexpr auto value = node_type::table;
	};
	template <>
	struct node_type_getter<array>
	{
		static constexpr auto value = node_type::array;
	};
	template <>
	struct node_type_getter<void>
	{
		static constexpr auto value = node_type::none;
	};
	template <typename T>
	inline constexpr node_type node_type_of = node_type_getter<unwrap_node<remove_cvref<T>>>::value;

	template <typename T, typename ConvertFrom>
	inline constexpr bool is_constructible_or_convertible = std::is_constructible_v<T, ConvertFrom> //
														 || std::is_convertible_v<ConvertFrom, T>;
}
TOML_IMPL_NAMESPACE_END;
/// \endcond

TOML_NAMESPACE_START
{
	/// \brief	Metafunction for determining if a type is, or is a reference to, a toml::table.
	template <typename T>
	inline constexpr bool is_table = std::is_same_v<impl::remove_cvref<T>, table>;

	/// \brief	Metafunction for determining if a type is, or is a reference to, a toml::array.
	template <typename T>
	inline constexpr bool is_array = std::is_same_v<impl::remove_cvref<T>, array>;

	/// \brief	Metafunction for determining if a type satisfies either toml::is_table or toml::is_array.
	template <typename T>
	inline constexpr bool is_container = is_table<T> || is_array<T>;

	/// \brief	Metafunction for determining if a type is, or is a reference to, a std::string or toml::value<std::string>.
	template <typename T>
	inline constexpr bool is_string = std::is_same_v<				//
		impl::remove_cvref<impl::wrap_node<impl::remove_cvref<T>>>, //
		value<std::string>>;

	/// \brief	Metafunction for determining if a type is, or is a reference to, a int64_t or toml::value<int64_t>.
	template <typename T>
	inline constexpr bool is_integer = std::is_same_v<				//
		impl::remove_cvref<impl::wrap_node<impl::remove_cvref<T>>>, //
		value<int64_t>>;

	/// \brief	Metafunction for determining if a type is, or is a reference to, a double or toml::value<double>.
	template <typename T>
	inline constexpr bool is_floating_point = std::is_same_v<		//
		impl::remove_cvref<impl::wrap_node<impl::remove_cvref<T>>>, //
		value<double>>;

	/// \brief	Metafunction for determining if a type satisfies either toml::is_integer or toml::is_floating_point.
	template <typename T>
	inline constexpr bool is_number = is_integer<T> || is_floating_point<T>;

	/// \brief	Metafunction for determining if a type is, or is a reference to, a bool or toml::value<bool>.
	template <typename T>
	inline constexpr bool is_boolean = std::is_same_v<				//
		impl::remove_cvref<impl::wrap_node<impl::remove_cvref<T>>>, //
		value<bool>>;

	/// \brief	Metafunction for determining if a type is, or is a reference to, a toml::date or toml::value<date>.
	template <typename T>
	inline constexpr bool is_date = std::is_same_v<					//
		impl::remove_cvref<impl::wrap_node<impl::remove_cvref<T>>>, //
		value<date>>;

	/// \brief	Metafunction for determining if a type is, or is a reference to, a toml::time or toml::value<time>.
	template <typename T>
	inline constexpr bool is_time = std::is_same_v<					//
		impl::remove_cvref<impl::wrap_node<impl::remove_cvref<T>>>, //
		value<time>>;

	/// \brief	Metafunction for determining if a type is, or is a reference to, a toml::date_time or toml::value<date_time>.
	template <typename T>
	inline constexpr bool is_date_time = std::is_same_v<			//
		impl::remove_cvref<impl::wrap_node<impl::remove_cvref<T>>>, //
		value<date_time>>;

	/// \brief	Metafunction for determining if a type satisfies any of toml::is_date, toml::is_time or toml::is_date_time.
	template <typename T>
	inline constexpr bool is_chronological = is_date<T> || is_time<T> || is_date_time<T>;

	/// \brief	Metafunction for determining if a type is, or is a reference to, any of the toml value types. Excludes tables and arrays.
	template <typename T>
	inline constexpr bool is_value = is_string<T> || is_number<T> || is_boolean<T> || is_chronological<T>;

	/// \brief	Metafunction for determining if a type is, or is a reference to, a toml::node (or one of its subclasses).
	template <typename T>
	inline constexpr bool is_node = std::is_same_v<toml::node, impl::remove_cvref<T>> //
								 || std::is_base_of_v<toml::node, impl::remove_cvref<T>>;

	/// \brief	Metafunction for determining if a type is, or is a reference to, a toml::node_view.
	template <typename T>
	inline constexpr bool is_node_view = impl::is_one_of<impl::remove_cvref<T>, node_view<node>, node_view<const node>>;
}
TOML_NAMESPACE_END;

//#---------------------------------------------------------------------------------------------------------------------
//# INTERNAL HELPERS
//#---------------------------------------------------------------------------------------------------------------------
/// \cond
TOML_IMPL_NAMESPACE_START
{
	template <typename T>
	TOML_CONST_INLINE_GETTER
	constexpr std::underlying_type_t<T> unwrap_enum(T val) noexcept
	{
		return static_cast<std::underlying_type_t<T>>(val);
	}

	// Q: "why not use std::fpclassify?"
	// A: Because it gets broken by -ffast-math and friends
	enum class TOML_CLOSED_ENUM fp_class : unsigned
	{
		ok,
		neg_inf,
		pos_inf,
		nan
	};

	TOML_PURE_GETTER
	inline fp_class fpclassify(const double& val) noexcept
	{
		static_assert(sizeof(uint64_t) == sizeof(double));

		static constexpr uint64_t sign	   = 0b1000000000000000000000000000000000000000000000000000000000000000ull;
		static constexpr uint64_t exponent = 0b0111111111110000000000000000000000000000000000000000000000000000ull;
		static constexpr uint64_t mantissa = 0b0000000000001111111111111111111111111111111111111111111111111111ull;

		uint64_t val_bits;
		std::memcpy(&val_bits, &val, sizeof(val));
		if ((val_bits & exponent) != exponent)
			return fp_class::ok;
		if ((val_bits & mantissa))
			return fp_class::nan;
		return (val_bits & sign) ? fp_class::neg_inf : fp_class::pos_inf;
	}

	// Q: "why not use std::find and std::min?"
	// A: Because <algorithm> is _huge_ and these would be the only things I used from it.
	//    I don't want to impose such a heavy compile-time burden on users.

	template <typename Iterator, typename T>
	TOML_PURE_GETTER
	inline auto find(Iterator start, Iterator end, const T& needle) noexcept //
		->decltype(&(*start))
	{
		for (; start != end; start++)
			if (*start == needle)
				return &(*start);
		return nullptr;
	}

	template <typename T>
	TOML_PURE_GETTER
	constexpr const T& min(const T& a, const T& b) noexcept //
	{
		return a < b ? a : b;
	}
}
TOML_IMPL_NAMESPACE_END;
/// \endcond

#include "header_end.h"
