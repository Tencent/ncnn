//# This file is a part of toml++ and is subject to the the terms of the MIT license.
//# Copyright (c) Mark Gillard <mark.gillard@outlook.com.au>
//# See https://github.com/marzer/tomlplusplus/blob/master/LICENSE for the full license text.
// SPDX-License-Identifier: MIT
#pragma once

#include "preprocessor.h"
//# {{
#if !TOML_IMPLEMENTATION
#error This is an implementation-only header.
#endif
//# }}
#if TOML_ENABLE_FORMATTERS

#include "formatter.h"
#include "print_to_stream.h"
#include "value.h"
#include "table.h"
#include "array.h"
#include "unicode.h"
#include "parse_result.h"
#include "header_start.h"

TOML_IMPL_NAMESPACE_START
{
	enum class TOML_CLOSED_FLAGS_ENUM formatted_string_traits : unsigned
	{
		none,
		line_breaks	  = 1u << 0, // \n
		tabs		  = 1u << 1, // \t
		control_chars = 1u << 2, // also includes non-ascii vertical whitespace
		single_quotes = 1u << 3,
		non_bare	  = 1u << 4, // anything not satisfying "is bare key character"
		non_ascii	  = 1u << 5, // any codepoint >= 128

		all = (non_ascii << 1u) - 1u
	};
	TOML_MAKE_FLAGS(formatted_string_traits);

	TOML_EXTERNAL_LINKAGE
	formatter::formatter(const node* source_node,
						 const parse_result* source_pr,
						 const formatter_constants& constants,
						 const formatter_config& config) noexcept //
#if TOML_ENABLE_PARSER && !TOML_EXCEPTIONS
		: source_{ source_pr && *source_pr ? &source_pr->table() : source_node },
		  result_{ source_pr },
#else
		: source_{ source_pr ? source_pr : source_node },
#endif
		  constants_{ &constants },
		  config_{ config }
	{
		TOML_ASSERT_ASSUME(source_);

		config_.flags = (config_.flags | constants_->mandatory_flags) & ~constants_->ignored_flags;

		indent_columns_ = {};
		for (auto c : config_.indent)
			indent_columns_ += c == '\t' ? 4u : 1u;

		int_format_mask_ = config_.flags
						 & (format_flags::allow_binary_integers | format_flags::allow_octal_integers
							| format_flags::allow_hexadecimal_integers);
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::attach(std::ostream & stream) noexcept
	{
		indent_		   = {};
		naked_newline_ = true;
		stream_		   = &stream;
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::detach() noexcept
	{
		stream_ = nullptr;
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print_newline(bool force)
	{
		if (!naked_newline_ || force)
		{
			print_to_stream(*stream_, '\n');
			naked_newline_ = true;
		}
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print_indent()
	{
		for (int i = 0; i < indent_; i++)
		{
			print_to_stream(*stream_, config_.indent);
			naked_newline_ = false;
		}
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print_unformatted(char c)
	{
		print_to_stream(*stream_, c);
		naked_newline_ = false;
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print_unformatted(std::string_view str)
	{
		print_to_stream(*stream_, str);
		naked_newline_ = false;
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print_string(std::string_view str, bool allow_multi_line, bool allow_bare)
	{
		if (str.empty())
		{
			print_unformatted(literal_strings_allowed() ? "''"sv : "\"\""sv);
			return;
		}

		// pre-scan the string to determine how we should output it
		formatted_string_traits traits = {};

		if (!allow_bare)
			traits |= formatted_string_traits::non_bare;
		bool unicode_allowed = unicode_strings_allowed();

		// ascii fast path
		if (is_ascii(str.data(), str.length()))
		{
			for (auto c : str)
			{
				switch (c)
				{
					case '\n': traits |= formatted_string_traits::line_breaks; break;
					case '\t': traits |= formatted_string_traits::tabs; break;
					case '\'': traits |= formatted_string_traits::single_quotes; break;
					default:
					{
						if TOML_UNLIKELY(is_control_character(c))
							traits |= formatted_string_traits::control_chars;

						if (!is_ascii_bare_key_character(static_cast<char32_t>(c)))
							traits |= formatted_string_traits::non_bare;
						break;
					}
				}

				static constexpr auto all_ascii_traits =
					formatted_string_traits::all & ~formatted_string_traits::non_ascii;
				if (traits == all_ascii_traits)
					break;
			}
		}

		// unicode slow path
		else
		{
			traits |= formatted_string_traits::non_ascii;
			utf8_decoder decoder;

			// if the unicode is malformed just treat the string as a single-line non-literal and
			// escape all non-ascii characters (to ensure round-tripping and help with diagnostics)
			const auto bad_unicode = [&]() noexcept
			{
				traits &= ~formatted_string_traits::line_breaks;
				traits |= formatted_string_traits::control_chars | formatted_string_traits::non_bare;
				unicode_allowed = false;
			};

			for (auto c : str)
			{
				decoder(c);

				if TOML_UNLIKELY(decoder.error())
				{
					bad_unicode();
					break;
				}

				if (!decoder.has_code_point())
					continue;

				switch (decoder.codepoint)
				{
					case U'\n': traits |= formatted_string_traits::line_breaks; break;
					case U'\t': traits |= formatted_string_traits::tabs; break;
					case U'\'': traits |= formatted_string_traits::single_quotes; break;
					default:
					{
						if TOML_UNLIKELY(is_control_character(decoder.codepoint)
							|| is_non_ascii_vertical_whitespace(decoder.codepoint))
							traits |= formatted_string_traits::control_chars;

						if (!is_bare_key_character(decoder.codepoint))
							traits |= formatted_string_traits::non_bare;
						break;
					}
				}
			}

			if (decoder.needs_more_input())
				bad_unicode();
		}

		// if the string meets the requirements of being 'bare' we can emit a bare string
		// (bare strings are composed of letters and numbers; no whitespace, control chars, quotes, etc)
		if (!(traits & formatted_string_traits::non_bare)
			&& (!(traits & formatted_string_traits::non_ascii) || unicode_allowed))
		{
			print_unformatted(str);
			return;
		}

		// determine if this should be a multi-line string (triple-quotes)
		const auto multi_line = allow_multi_line			 //
							 && multi_line_strings_allowed() //
							 && !!(traits & formatted_string_traits::line_breaks);

		// determine if this should be a literal string (single-quotes with no escaping)
		const auto literal = literal_strings_allowed()													   //
						  && !(traits & formatted_string_traits::control_chars)							   //
						  && (!(traits & formatted_string_traits::single_quotes) || multi_line)			   //
						  && (!(traits & formatted_string_traits::tabs) || real_tabs_in_strings_allowed()) //
						  && (!(traits & formatted_string_traits::non_ascii) || unicode_allowed);

		// literal strings (single quotes, no escape codes)
		if (literal)
		{
			const auto quot = multi_line ? R"(''')"sv : R"(')"sv;
			print_unformatted(quot);
			print_unformatted(str);
			print_unformatted(quot);
			return;
		}

		// anything from here down is a non-literal string, so requires iteration and escaping.
		print_unformatted(multi_line ? R"(""")"sv : R"(")"sv);

		const auto real_tabs_allowed = real_tabs_in_strings_allowed();

		// ascii fast path
		if (!(traits & formatted_string_traits::non_ascii))
		{
			for (auto c : str)
			{
				switch (c)
				{
					case '"': print_to_stream(*stream_, R"(\")"sv); break;
					case '\\': print_to_stream(*stream_, R"(\\)"sv); break;
					case '\x7F': print_to_stream(*stream_, R"(\u007F)"sv); break;
					case '\t': print_to_stream(*stream_, real_tabs_allowed ? "\t"sv : R"(\t)"sv); break;
					case '\n': print_to_stream(*stream_, multi_line ? "\n"sv : R"(\n)"sv); break;
					default:
					{
						// control characters from lookup table
						if TOML_UNLIKELY(c >= '\x00' && c <= '\x1F')
							print_to_stream(*stream_, control_char_escapes[c]);

						// regular characters
						else
							print_to_stream(*stream_, c);
					}
				}
			}
		}

		// unicode slow path
		else
		{
			utf8_decoder decoder;
			const char* cp_start = str.data();
			const char* cp_end	 = cp_start;
			for (auto c : str)
			{
				decoder(c);
				cp_end++;

				// if the decoder encounters malformed unicode just emit raw bytes and
				if (decoder.error())
				{
					while (cp_start != cp_end)
					{
						print_to_stream(*stream_, R"(\u00)"sv);
						print_to_stream(*stream_,
										static_cast<uint8_t>(*cp_start),
										value_flags::format_as_hexadecimal,
										2);
						cp_start++;
					}
					decoder.reset();
					continue;
				}

				if (!decoder.has_code_point())
					continue;

				switch (decoder.codepoint)
				{
					case U'"': print_to_stream(*stream_, R"(\")"sv); break;
					case U'\\': print_to_stream(*stream_, R"(\\)"sv); break;
					case U'\x7F': print_to_stream(*stream_, R"(\u007F)"sv); break;
					case U'\t': print_to_stream(*stream_, real_tabs_allowed ? "\t"sv : R"(\t)"sv); break;
					case U'\n': print_to_stream(*stream_, multi_line ? "\n"sv : R"(\n)"sv); break;
					default:
					{
						// control characters from lookup table
						if TOML_UNLIKELY(decoder.codepoint <= U'\x1F')
							print_to_stream(*stream_,
											control_char_escapes[static_cast<uint_least32_t>(decoder.codepoint)]);

						// escaped unicode characters
						else if (decoder.codepoint > U'\x7F'
								 && (!unicode_allowed || is_non_ascii_vertical_whitespace(decoder.codepoint)))
						{
							if (static_cast<uint_least32_t>(decoder.codepoint) > 0xFFFFu)
							{
								print_to_stream(*stream_, R"(\U)"sv);
								print_to_stream(*stream_,
												static_cast<uint_least32_t>(decoder.codepoint),
												value_flags::format_as_hexadecimal,
												8);
							}
							else
							{
								print_to_stream(*stream_, R"(\u)"sv);
								print_to_stream(*stream_,
												static_cast<uint_least32_t>(decoder.codepoint),
												value_flags::format_as_hexadecimal,
												4);
							}
						}

						// regular characters
						else
							print_to_stream(*stream_, cp_start, static_cast<size_t>(cp_end - cp_start));
					}
				}

				cp_start = cp_end;
			}
		}

		print_unformatted(multi_line ? R"(""")"sv : R"(")"sv);
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print(const value<std::string>& val)
	{
		print_string(val.get());
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print(const value<int64_t>& val)
	{
		naked_newline_ = false;

		if (*val >= 0 && !!int_format_mask_)
		{
			static constexpr auto value_flags_mask =
				value_flags::format_as_binary | value_flags::format_as_octal | value_flags::format_as_hexadecimal;

			const auto fmt = val.flags() & value_flags_mask;
			switch (fmt)
			{
				case value_flags::format_as_binary:
					if (!!(int_format_mask_ & format_flags::allow_binary_integers))
					{
						print_to_stream(*stream_, "0b"sv);
						print_to_stream(*stream_, *val, fmt);
						return;
					}
					break;

				case value_flags::format_as_octal:
					if (!!(int_format_mask_ & format_flags::allow_octal_integers))
					{
						print_to_stream(*stream_, "0o"sv);
						print_to_stream(*stream_, *val, fmt);
						return;
					}
					break;

				case value_flags::format_as_hexadecimal:
					if (!!(int_format_mask_ & format_flags::allow_hexadecimal_integers))
					{
						print_to_stream(*stream_, "0x"sv);
						print_to_stream(*stream_, *val, fmt);
						return;
					}
					break;

				default: break;
			}
		}

		// fallback to decimal
		print_to_stream(*stream_, *val);
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print(const value<double>& val)
	{
		const std::string_view* inf_nan = nullptr;
		switch (fpclassify(*val))
		{
			case fp_class::neg_inf: inf_nan = &constants_->float_neg_inf; break;
			case fp_class::pos_inf: inf_nan = &constants_->float_pos_inf; break;
			case fp_class::nan: inf_nan = &constants_->float_nan; break;
			case fp_class::ok:
				print_to_stream(*stream_,
								*val,
								value_flags::none,
								!!(config_.flags & format_flags::relaxed_float_precision));
				break;
			default: TOML_UNREACHABLE;
		}

		if (inf_nan)
		{
			if (!!(config_.flags & format_flags::quote_infinities_and_nans))
				print_to_stream_bookended(*stream_, *inf_nan, '"');
			else
				print_to_stream(*stream_, *inf_nan);
		}

		naked_newline_ = false;
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print(const value<bool>& val)
	{
		print_unformatted(*val ? constants_->bool_true : constants_->bool_false);
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print(const value<date>& val)
	{
		if (!!(config_.flags & format_flags::quote_dates_and_times))
			print_to_stream_bookended(*stream_, *val, literal_strings_allowed() ? '\'' : '"');
		else
			print_to_stream(*stream_, *val);
		naked_newline_ = false;
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print(const value<time>& val)
	{
		if (!!(config_.flags & format_flags::quote_dates_and_times))
			print_to_stream_bookended(*stream_, *val, literal_strings_allowed() ? '\'' : '"');
		else
			print_to_stream(*stream_, *val);
		naked_newline_ = false;
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print(const value<date_time>& val)
	{
		if (!!(config_.flags & format_flags::quote_dates_and_times))
			print_to_stream_bookended(*stream_, *val, literal_strings_allowed() ? '\'' : '"');
		else
			print_to_stream(*stream_, *val);
		naked_newline_ = false;
	}

	TOML_EXTERNAL_LINKAGE
	void formatter::print_value(const node& val_node, node_type type)
	{
		TOML_ASSUME(type > node_type::array);
		switch (type)
		{
			case node_type::string: print(*reinterpret_cast<const value<std::string>*>(&val_node)); break;
			case node_type::integer: print(*reinterpret_cast<const value<int64_t>*>(&val_node)); break;
			case node_type::floating_point: print(*reinterpret_cast<const value<double>*>(&val_node)); break;
			case node_type::boolean: print(*reinterpret_cast<const value<bool>*>(&val_node)); break;
			case node_type::date: print(*reinterpret_cast<const value<date>*>(&val_node)); break;
			case node_type::time: print(*reinterpret_cast<const value<time>*>(&val_node)); break;
			case node_type::date_time: print(*reinterpret_cast<const value<date_time>*>(&val_node)); break;
			default: TOML_UNREACHABLE;
		}
	}

#if TOML_ENABLE_PARSER && !TOML_EXCEPTIONS

	TOML_EXTERNAL_LINKAGE
	bool formatter::dump_failed_parse_result()
	{
		if (result_ && !(*result_))
		{
			stream() << result_->error();
			return true;
		}
		return false;
	}

#else

	TOML_EXTERNAL_LINKAGE
	TOML_ATTR(const)
	bool formatter::dump_failed_parse_result()
	{
		return false;
	}

#endif
}
TOML_IMPL_NAMESPACE_END;

#include "header_end.h"
#endif // TOML_ENABLE_FORMATTERS
