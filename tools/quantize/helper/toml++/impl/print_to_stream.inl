//# This file is a part of toml++ and is subject to the the terms of the MIT license.
//# Copyright (c) Mark Gillard <mark.gillard@outlook.com.au>
//# See https://github.com/marzer/tomlplusplus/blob/master/LICENSE for the full license text.
// SPDX-License-Identifier: MIT
#pragma once

//# {{
#include "preprocessor.h"
#if !TOML_IMPLEMENTATION
#error This is an implementation-only header.
#endif
//# }}

#include "print_to_stream.h"
#include "source_region.h"
#include "date_time.h"
#include "toml_formatter.h"
#include "value.h"
#include "array.h"
#include "table.h"
TOML_DISABLE_WARNINGS;
#include <ostream>
#if TOML_INT_CHARCONV || TOML_FLOAT_CHARCONV
#include <charconv>
#endif
#if !TOML_INT_CHARCONV || !TOML_FLOAT_CHARCONV
#include <sstream>
#endif
#if !TOML_INT_CHARCONV
#include <iomanip>
#endif
TOML_ENABLE_WARNINGS;
#include "header_start.h"

TOML_ANON_NAMESPACE_START
{
	template <typename T>
	inline constexpr size_t charconv_buffer_length = 0;

	template <>
	inline constexpr size_t charconv_buffer_length<int8_t> = 4; // strlen("-128")

	template <>
	inline constexpr size_t charconv_buffer_length<int16_t> = 6; // strlen("-32768")

	template <>
	inline constexpr size_t charconv_buffer_length<int32_t> = 11; // strlen("-2147483648")

	template <>
	inline constexpr size_t charconv_buffer_length<int64_t> = 20; // strlen("-9223372036854775808")

	template <>
	inline constexpr size_t charconv_buffer_length<uint8_t> = 3; // strlen("255")

	template <>
	inline constexpr size_t charconv_buffer_length<uint16_t> = 5; // strlen("65535")

	template <>
	inline constexpr size_t charconv_buffer_length<uint32_t> = 10; // strlen("4294967295")

	template <>
	inline constexpr size_t charconv_buffer_length<uint64_t> = 20; // strlen("18446744073709551615")

	template <>
	inline constexpr size_t charconv_buffer_length<float> = 64;

	template <>
	inline constexpr size_t charconv_buffer_length<double> = 64;

	template <typename T>
	TOML_INTERNAL_LINKAGE
	void print_integer_to_stream(std::ostream & stream, T val, value_flags format = {}, size_t min_digits = 0)
	{
		if (!val)
		{
			if (!min_digits)
				min_digits = 1;

			for (size_t i = 0; i < min_digits; i++)
				stream.put('0');

			return;
		}

		static constexpr auto value_flags_mask =
			value_flags::format_as_binary | value_flags::format_as_octal | value_flags::format_as_hexadecimal;
		format &= value_flags_mask;

		int base = 10;
		if (format != value_flags::none && val > T{})
		{
			switch (format)
			{
				case value_flags::format_as_binary: base = 2; break;
				case value_flags::format_as_octal: base = 8; break;
				case value_flags::format_as_hexadecimal: base = 16; break;
				default: break;
			}
		}

#if TOML_INT_CHARCONV

		char buf[(sizeof(T) * CHAR_BIT)];
		const auto res = std::to_chars(buf, buf + sizeof(buf), val, base);
		const auto len = static_cast<size_t>(res.ptr - buf);
		for (size_t i = len; i < min_digits; i++)
			stream.put('0');
		if (base == 16)
		{
			for (size_t i = 0; i < len; i++)
				if (buf[i] >= 'a')
					buf[i] -= 32;
		}
		impl::print_to_stream(stream, buf, len);

#else

		using unsigned_type = std::conditional_t<(sizeof(T) > sizeof(unsigned)), std::make_unsigned_t<T>, unsigned>;
		using cast_type		= std::conditional_t<std::is_signed_v<T>, std::make_signed_t<unsigned_type>, unsigned_type>;

		if (base == 2)
		{
			const auto len = sizeof(T) * CHAR_BIT;
			for (size_t i = len; i < min_digits; i++)
				stream.put('0');

			bool found_one	   = false;
			const auto v	   = static_cast<unsigned_type>(val);
			unsigned_type mask = unsigned_type{ 1 } << (len - 1u);
			for (size_t i = 0; i < len; i++)
			{
				if ((v & mask))
				{
					stream.put('1');
					found_one = true;
				}
				else if (found_one)
					stream.put('0');
				mask >>= 1;
			}
		}
		else
		{
			std::ostringstream ss;
			ss.imbue(std::locale::classic());
			ss << std::uppercase << std::setbase(base);
			if (min_digits)
				ss << std::setfill('0') << std::setw(static_cast<int>(min_digits));
			ss << static_cast<cast_type>(val);
			const auto str = std::move(ss).str();
			impl::print_to_stream(stream, str);
		}

#endif
	}

	template <typename T>
	TOML_INTERNAL_LINKAGE
	void print_floating_point_to_stream(std::ostream & stream,
										T val,
										value_flags format,
										[[maybe_unused]] bool relaxed_precision)
	{
		switch (impl::fpclassify(val))
		{
			case impl::fp_class::neg_inf: impl::print_to_stream(stream, "-inf"sv); break;

			case impl::fp_class::pos_inf: impl::print_to_stream(stream, "inf"sv); break;

			case impl::fp_class::nan: impl::print_to_stream(stream, "nan"sv); break;

			case impl::fp_class::ok:
			{
				static constexpr auto needs_decimal_point = [](auto&& s) noexcept
				{
					for (auto c : s)
						if (c == '.' || c == 'E' || c == 'e')
							return false;
					return true;
				};

#if TOML_FLOAT_CHARCONV

				const auto hex = !!(format & value_flags::format_as_hexadecimal);
				char buf[charconv_buffer_length<T>];
				auto res = hex ? std::to_chars(buf, buf + sizeof(buf), val, std::chars_format::hex)
							   : std::to_chars(buf, buf + sizeof(buf), val);
				auto str = std::string_view{ buf, static_cast<size_t>(res.ptr - buf) };

				char buf2[charconv_buffer_length<T>];
				if (!hex && relaxed_precision)
				{
					res				= std::to_chars(buf2, buf2 + sizeof(buf2), val, std::chars_format::general, 6);
					const auto str2 = std::string_view{ buf2, static_cast<size_t>(res.ptr - buf2) };
					if (str2.length() < str.length())
						str = str2;
				}

				impl::print_to_stream(stream, str);
				if (!hex && needs_decimal_point(str))
					toml::impl::print_to_stream(stream, ".0"sv);

#else

				std::ostringstream ss;
				ss.imbue(std::locale::classic());
				if (!relaxed_precision)
					ss.precision(std::numeric_limits<T>::max_digits10);
				if (!!(format & value_flags::format_as_hexadecimal))
					ss << std::hexfloat;
				ss << val;
				const auto str = std::move(ss).str();
				impl::print_to_stream(stream, str);
				if (!(format & value_flags::format_as_hexadecimal) && needs_decimal_point(str))
					impl::print_to_stream(stream, ".0"sv);

#endif
			}
			break;

			default: TOML_UNREACHABLE;
		}
	}
}
TOML_ANON_NAMESPACE_END;

TOML_IMPL_NAMESPACE_START
{
	TOML_EXTERNAL_LINKAGE
	TOML_ATTR(nonnull)
	void print_to_stream(std::ostream & stream, const char* val, size_t len)
	{
		stream.write(val, static_cast<std::streamsize>(len));
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, std::string_view val)
	{
		stream.write(val.data(), static_cast<std::streamsize>(val.length()));
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const std::string& val)
	{
		stream.write(val.data(), static_cast<std::streamsize>(val.length()));
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, char val)
	{
		stream.put(val);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, int8_t val, value_flags format, size_t min_digits)
	{
		TOML_ANON_NAMESPACE::print_integer_to_stream(stream, val, format, min_digits);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, int16_t val, value_flags format, size_t min_digits)
	{
		TOML_ANON_NAMESPACE::print_integer_to_stream(stream, val, format, min_digits);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, int32_t val, value_flags format, size_t min_digits)
	{
		TOML_ANON_NAMESPACE::print_integer_to_stream(stream, val, format, min_digits);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, int64_t val, value_flags format, size_t min_digits)
	{
		TOML_ANON_NAMESPACE::print_integer_to_stream(stream, val, format, min_digits);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, uint8_t val, value_flags format, size_t min_digits)
	{
		TOML_ANON_NAMESPACE::print_integer_to_stream(stream, val, format, min_digits);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, uint16_t val, value_flags format, size_t min_digits)
	{
		TOML_ANON_NAMESPACE::print_integer_to_stream(stream, val, format, min_digits);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, uint32_t val, value_flags format, size_t min_digits)
	{
		TOML_ANON_NAMESPACE::print_integer_to_stream(stream, val, format, min_digits);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, uint64_t val, value_flags format, size_t min_digits)
	{
		TOML_ANON_NAMESPACE::print_integer_to_stream(stream, val, format, min_digits);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, float val, value_flags format, bool relaxed_precision)
	{
		TOML_ANON_NAMESPACE::print_floating_point_to_stream(stream, val, format, relaxed_precision);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, double val, value_flags format, bool relaxed_precision)
	{
		TOML_ANON_NAMESPACE::print_floating_point_to_stream(stream, val, format, relaxed_precision);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, bool val)
	{
		print_to_stream(stream, val ? "true"sv : "false"sv);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const toml::date& val)
	{
		print_to_stream(stream, val.year, {}, 4);
		stream.put('-');
		print_to_stream(stream, val.month, {}, 2);
		stream.put('-');
		print_to_stream(stream, val.day, {}, 2);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const toml::time& val)
	{
		print_to_stream(stream, val.hour, {}, 2);
		stream.put(':');
		print_to_stream(stream, val.minute, {}, 2);
		stream.put(':');
		print_to_stream(stream, val.second, {}, 2);
		if (val.nanosecond && val.nanosecond <= 999999999u)
		{
			stream.put('.');
			auto ns		  = val.nanosecond;
			size_t digits = 9u;
			while (ns % 10u == 0u)
			{
				ns /= 10u;
				digits--;
			}
			print_to_stream(stream, ns, {}, digits);
		}
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const toml::time_offset& val)
	{
		if (!val.minutes)
		{
			stream.put('Z');
			return;
		}

		auto mins = static_cast<int>(val.minutes);
		if (mins < 0)
		{
			stream.put('-');
			mins = -mins;
		}
		else
			stream.put('+');
		const auto hours = mins / 60;
		if (hours)
		{
			print_to_stream(stream, static_cast<unsigned int>(hours), {}, 2);
			mins -= hours * 60;
		}
		else
			print_to_stream(stream, "00"sv);
		stream.put(':');
		print_to_stream(stream, static_cast<unsigned int>(mins), {}, 2);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const toml::date_time& val)
	{
		print_to_stream(stream, val.date);
		stream.put('T');
		print_to_stream(stream, val.time);
		if (val.offset)
			print_to_stream(stream, *val.offset);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const source_position& val)
	{
		print_to_stream(stream, "line "sv);
		print_to_stream(stream, val.line);
		print_to_stream(stream, ", column "sv);
		print_to_stream(stream, val.column);
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const source_region& val)
	{
		print_to_stream(stream, val.begin);
		if (val.path)
		{
			print_to_stream(stream, " of '"sv);
			print_to_stream(stream, *val.path);
			stream.put('\'');
		}
	}

#if TOML_ENABLE_FORMATTERS

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const array& arr)
	{
		stream << toml_formatter{ arr };
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const table& tbl)
	{
		stream << toml_formatter{ tbl };
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const value<std::string>& val)
	{
		stream << toml_formatter{ val };
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const value<int64_t>& val)
	{
		stream << toml_formatter{ val };
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const value<double>& val)
	{
		stream << toml_formatter{ val };
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const value<bool>& val)
	{
		stream << toml_formatter{ val };
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const value<date>& val)
	{
		stream << toml_formatter{ val };
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const value<time>& val)
	{
		stream << toml_formatter{ val };
	}

	TOML_EXTERNAL_LINKAGE
	void print_to_stream(std::ostream & stream, const value<date_time>& val)
	{
		stream << toml_formatter{ val };
	}

#endif
}
TOML_IMPL_NAMESPACE_END;

#include "header_end.h"
