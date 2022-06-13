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

#include "toml_formatter.h"
#include "print_to_stream.h"
#include "value.h"
#include "table.h"
#include "array.h"
#include "unicode.h"
#include "header_start.h"
TOML_DISABLE_ARITHMETIC_WARNINGS;

TOML_ANON_NAMESPACE_START
{
	TOML_INTERNAL_LINKAGE
	size_t toml_formatter_count_inline_columns(const node& node, size_t line_wrap_cols) noexcept
	{
		switch (node.type())
		{
			case node_type::table:
			{
				auto& tbl = *reinterpret_cast<const table*>(&node);
				if (tbl.empty())
					return 2u;		// "{}"
				size_t weight = 3u; // "{ }"
				for (auto&& [k, v] : tbl)
				{
					weight += k.length() + toml_formatter_count_inline_columns(v, line_wrap_cols) + 2u; // +  ", "
					if (weight >= line_wrap_cols)
						break;
				}
				return weight;
			}

			case node_type::array:
			{
				auto& arr = *reinterpret_cast<const array*>(&node);
				if (arr.empty())
					return 2u;		// "[]"
				size_t weight = 3u; // "[ ]"
				for (auto& elem : arr)
				{
					weight += toml_formatter_count_inline_columns(elem, line_wrap_cols) + 2u; // +  ", "
					if (weight >= line_wrap_cols)
						break;
				}
				return weight;
			}

			case node_type::string:
			{
				// todo: proper utf8 decoding?
				// todo: tab awareness?
				auto& str = (*reinterpret_cast<const value<std::string>*>(&node)).get();
				return str.length() + 2u; // + ""
			}

			case node_type::integer:
			{
				auto val = (*reinterpret_cast<const value<int64_t>*>(&node)).get();
				if (!val)
					return 1u;
				size_t weight = {};
				if (val < 0)
				{
					weight += 1u;
					val *= -1;
				}
				return weight + static_cast<size_t>(log10(static_cast<double>(val))) + 1u;
			}

			case node_type::floating_point:
			{
				auto val = (*reinterpret_cast<const value<double>*>(&node)).get();
				if (val == 0.0)
					return 3u;		// "0.0"
				size_t weight = 2u; // ".0"
				if (val < 0.0)
				{
					weight += 1u;
					val *= -1.0;
				}
				return weight + static_cast<size_t>(log10(val)) + 1u;
				break;
			}

			case node_type::boolean: return 5u;
			case node_type::date: [[fallthrough]];
			case node_type::time: return 10u;
			case node_type::date_time: return 30u;
			case node_type::none: TOML_UNREACHABLE;
			default: TOML_UNREACHABLE;
		}

		TOML_UNREACHABLE;
	}

	TOML_INTERNAL_LINKAGE
	bool toml_formatter_forces_multiline(const node& node, size_t line_wrap_cols, size_t starting_column_bias) noexcept
	{
		return (toml_formatter_count_inline_columns(node, line_wrap_cols) + starting_column_bias) >= line_wrap_cols;
	}
}
TOML_ANON_NAMESPACE_END;

TOML_NAMESPACE_START
{
	TOML_EXTERNAL_LINKAGE
	void toml_formatter::print_pending_table_separator()
	{
		if (pending_table_separator_)
		{
			print_newline(true);
			print_newline(true);
			pending_table_separator_ = false;
		}
	}

	TOML_EXTERNAL_LINKAGE
	void toml_formatter::print(const key& k)
	{
		print_string(k.str(), false, true);
	}

	TOML_EXTERNAL_LINKAGE
	void toml_formatter::print_inline(const table& tbl)
	{
		if (tbl.empty())
		{
			print_unformatted("{}"sv);
			return;
		}

		print_unformatted("{ "sv);

		bool first = false;
		for (auto&& [k, v] : tbl)
		{
			if (first)
				print_unformatted(", "sv);
			first = true;

			print(k);
			print_unformatted(" = "sv);

			const auto type = v.type();
			TOML_ASSUME(type != node_type::none);
			switch (type)
			{
				case node_type::table: print_inline(*reinterpret_cast<const table*>(&v)); break;
				case node_type::array: print(*reinterpret_cast<const array*>(&v)); break;
				default: print_value(v, type);
			}
		}

		print_unformatted(" }"sv);
	}

	TOML_EXTERNAL_LINKAGE
	void toml_formatter::print(const array& arr)
	{
		if (arr.empty())
		{
			print_unformatted("[]"sv);
			return;
		}

		const auto original_indent = indent();
		const auto multiline	   = TOML_ANON_NAMESPACE::toml_formatter_forces_multiline(
			  arr,
			  120u,
			  indent_columns() * static_cast<size_t>(original_indent < 0 ? 0 : original_indent));

		print_unformatted("["sv);

		if (multiline)
		{
			if (original_indent < 0)
				indent(0);
			if (indent_array_elements())
				increase_indent();
		}
		else
			print_unformatted(' ');

		for (size_t i = 0; i < arr.size(); i++)
		{
			if (i > 0u)
			{
				print_unformatted(',');
				if (!multiline)
					print_unformatted(' ');
			}

			if (multiline)
			{
				print_newline(true);
				print_indent();
			}

			auto& v			= arr[i];
			const auto type = v.type();
			TOML_ASSUME(type != node_type::none);
			switch (type)
			{
				case node_type::table: print_inline(*reinterpret_cast<const table*>(&v)); break;
				case node_type::array: print(*reinterpret_cast<const array*>(&v)); break;
				default: print_value(v, type);
			}
		}
		if (multiline)
		{
			indent(original_indent);
			print_newline(true);
			print_indent();
		}
		else
			print_unformatted(' ');

		print_unformatted("]"sv);
	}

	TOML_EXTERNAL_LINKAGE
	void toml_formatter::print(const table& tbl)
	{
		static constexpr auto is_non_inline_array_of_tables = [](auto&& nde) noexcept
		{
			auto arr = nde.as_array();
			return arr && arr->is_array_of_tables() && !arr->template get_as<table>(0u)->is_inline();
		};

		// values, arrays, and inline tables/table arrays
		for (auto&& [k, v] : tbl)
		{
			const auto type = v.type();
			if ((type == node_type::table && !reinterpret_cast<const table*>(&v)->is_inline())
				|| (type == node_type::array && is_non_inline_array_of_tables(v)))
				continue;

			pending_table_separator_ = true;
			print_newline();
			print_indent();
			print(k);
			print_unformatted(" = "sv);
			TOML_ASSUME(type != node_type::none);
			switch (type)
			{
				case node_type::table: print_inline(*reinterpret_cast<const table*>(&v)); break;
				case node_type::array: print(*reinterpret_cast<const array*>(&v)); break;
				default: print_value(v, type);
			}
		}

		const auto print_key_path = [&]()
		{
			size_t i{};
			for (const auto k : key_path_)
			{
				if (i++)
					print_unformatted('.');
				print(*k);
			}
		};

		// non-inline tables
		for (auto&& [k, v] : tbl)
		{
			const auto type = v.type();
			if (type != node_type::table || reinterpret_cast<const table*>(&v)->is_inline())
				continue;
			auto& child_tbl = *reinterpret_cast<const table*>(&v);

			// we can skip indenting and emitting the headers for tables that only contain other tables
			// (so we don't over-nest)
			size_t child_value_count{}; // includes inline tables and non-table arrays
			size_t child_table_count{};
			size_t child_table_array_count{};
			for (auto&& [child_k, child_v] : child_tbl)
			{
				TOML_UNUSED(child_k);
				const auto child_type = child_v.type();
				TOML_ASSUME(child_type != node_type::none);
				switch (child_type)
				{
					case node_type::table:
						if (reinterpret_cast<const table*>(&child_v)->is_inline())
							child_value_count++;
						else
							child_table_count++;
						break;

					case node_type::array:
						if (is_non_inline_array_of_tables(child_v))
							child_table_array_count++;
						else
							child_value_count++;
						break;

					default: child_value_count++;
				}
			}
			bool skip_self = false;
			if (child_value_count == 0u && (child_table_count > 0u || child_table_array_count > 0u))
				skip_self = true;

			key_path_.push_back(&k);

			if (!skip_self)
			{
				print_pending_table_separator();
				if (indent_sub_tables())
					increase_indent();
				print_indent();
				print_unformatted("["sv);
				print_key_path();
				print_unformatted("]"sv);
				pending_table_separator_ = true;
			}

			print(child_tbl);

			key_path_.pop_back();
			if (!skip_self && indent_sub_tables())
				decrease_indent();
		}

		// table arrays
		for (auto&& [k, v] : tbl)
		{
			if (!is_non_inline_array_of_tables(v))
				continue;
			auto& arr = *reinterpret_cast<const array*>(&v);

			if (indent_sub_tables())
				increase_indent();
			key_path_.push_back(&k);

			for (size_t i = 0; i < arr.size(); i++)
			{
				print_pending_table_separator();
				print_indent();
				print_unformatted("[["sv);
				print_key_path();
				print_unformatted("]]"sv);
				pending_table_separator_ = true;
				print(*reinterpret_cast<const table*>(&arr[i]));
			}

			key_path_.pop_back();
			if (indent_sub_tables())
				decrease_indent();
		}
	}

	TOML_EXTERNAL_LINKAGE
	void toml_formatter::print()
	{
		if (dump_failed_parse_result())
			return;

		switch (auto source_type = source().type())
		{
			case node_type::table:
			{
				auto& tbl = *reinterpret_cast<const table*>(&source());
				if (tbl.is_inline())
					print_inline(tbl);
				else
				{
					decrease_indent(); // so root kvps and tables have the same indent
					print(tbl);
				}
				break;
			}

			case node_type::array: print(*reinterpret_cast<const array*>(&source())); break;

			default: print_value(source(), source_type);
		}
	}
}
TOML_NAMESPACE_END;

#include "header_end.h"
#endif // TOML_ENABLE_FORMATTERS
