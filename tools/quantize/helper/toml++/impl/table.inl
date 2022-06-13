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

#include "table.h"
#include "node_view.h"
#include "header_start.h"

TOML_NAMESPACE_START
{
	TOML_EXTERNAL_LINKAGE
	table::table() noexcept
	{
#if TOML_LIFETIME_HOOKS
		TOML_TABLE_CREATED;
#endif
	}

	TOML_EXTERNAL_LINKAGE
	table::~table() noexcept
	{
#if TOML_LIFETIME_HOOKS
		TOML_TABLE_DESTROYED;
#endif
	}

	TOML_EXTERNAL_LINKAGE
	table::table(const impl::table_init_pair* b, const impl::table_init_pair* e)
	{
#if TOML_LIFETIME_HOOKS
		TOML_TABLE_CREATED;
#endif

		TOML_ASSERT_ASSUME(b);
		TOML_ASSERT_ASSUME(e);
		TOML_ASSERT_ASSUME(b <= e);

		if TOML_UNLIKELY(b == e)
			return;

		for (; b != e; b++)
		{
			if (!b->value) // empty node_views
				continue;

			map_.insert_or_assign(std::move(b->key), std::move(b->value));
		}
	}

	TOML_EXTERNAL_LINKAGE
	table::table(const table& other) //
		: node(other),
		  inline_{ other.inline_ }
	{
		for (auto&& [k, v] : other.map_)
			map_.emplace_hint(map_.end(), k, impl::make_node(*v));

#if TOML_LIFETIME_HOOKS
		TOML_TABLE_CREATED;
#endif
	}

	TOML_EXTERNAL_LINKAGE
	table::table(table && other) noexcept //
		: node(std::move(other)),
		  map_{ std::move(other.map_) },
		  inline_{ other.inline_ }
	{
#if TOML_LIFETIME_HOOKS
		TOML_TABLE_CREATED;
#endif
	}

	TOML_EXTERNAL_LINKAGE
	table& table::operator=(const table& rhs)
	{
		if (&rhs != this)
		{
			node::operator=(rhs);
			map_.clear();
			for (auto&& [k, v] : rhs.map_)
				map_.emplace_hint(map_.end(), k, impl::make_node(*v));
			inline_ = rhs.inline_;
		}
		return *this;
	}

	TOML_EXTERNAL_LINKAGE
	table& table::operator=(table&& rhs) noexcept
	{
		if (&rhs != this)
		{
			node::operator=(std::move(rhs));
			map_	= std::move(rhs.map_);
			inline_ = rhs.inline_;
		}
		return *this;
	}

	TOML_EXTERNAL_LINKAGE
	bool table::is_homogeneous(node_type ntype) const noexcept
	{
		if (map_.empty())
			return false;

		if (ntype == node_type::none)
			ntype = map_.cbegin()->second->type();

		for (auto&& [k, v] : map_)
		{
			TOML_UNUSED(k);
			if (v->type() != ntype)
				return false;
		}

		return true;
	}

	TOML_EXTERNAL_LINKAGE
	bool table::is_homogeneous(node_type ntype, node * &first_nonmatch) noexcept
	{
		if (map_.empty())
		{
			first_nonmatch = {};
			return false;
		}
		if (ntype == node_type::none)
			ntype = map_.cbegin()->second->type();
		for (const auto& [k, v] : map_)
		{
			TOML_UNUSED(k);
			if (v->type() != ntype)
			{
				first_nonmatch = v.get();
				return false;
			}
		}
		return true;
	}

	TOML_EXTERNAL_LINKAGE
	bool table::is_homogeneous(node_type ntype, const node*& first_nonmatch) const noexcept
	{
		node* fnm		  = nullptr;
		const auto result = const_cast<table&>(*this).is_homogeneous(ntype, fnm);
		first_nonmatch	  = fnm;
		return result;
	}

	TOML_EXTERNAL_LINKAGE
	node* table::get(std::string_view key) noexcept
	{
		if (auto it = map_.find(key); it != map_.end())
			return it->second.get();
		return nullptr;
	}

	TOML_EXTERNAL_LINKAGE
	node& table::at(std::string_view key)
	{
		auto n = get(key);

#if TOML_COMPILER_EXCEPTIONS

		if (!n)
		{
			auto err = "key '"s;
			err.append(key);
			err.append("' not found in table"sv);
			throw std::out_of_range{ err };
		}

#else

		TOML_ASSERT_ASSUME(n && "key not found in table!");

#endif

		return *n;
	}

	TOML_EXTERNAL_LINKAGE
	table::map_iterator table::get_lower_bound(std::string_view key) noexcept
	{
		return map_.lower_bound(key);
	}

	TOML_EXTERNAL_LINKAGE
	table::iterator table::find(std::string_view key) noexcept
	{
		return iterator{ map_.find(key) };
	}

	TOML_EXTERNAL_LINKAGE
	table::const_iterator table::find(std::string_view key) const noexcept
	{
		return const_iterator{ map_.find(key) };
	}

	TOML_EXTERNAL_LINKAGE
	table::map_iterator table::erase(const_map_iterator pos) noexcept
	{
		return map_.erase(pos);
	}

	TOML_EXTERNAL_LINKAGE
	table::map_iterator table::erase(const_map_iterator begin, const_map_iterator end) noexcept
	{
		return map_.erase(begin, end);
	}

	TOML_EXTERNAL_LINKAGE
	size_t table::erase(std::string_view key) noexcept
	{
		if (auto it = map_.find(key); it != map_.end())
		{
			map_.erase(it);
			return size_t{ 1 };
		}
		return size_t{};
	}

	TOML_EXTERNAL_LINKAGE
	table& table::prune(bool recursive)& noexcept
	{
		if (map_.empty())
			return *this;

		for (auto it = map_.begin(); it != map_.end();)
		{
			if (auto arr = it->second->as_array())
			{
				if (recursive)
					arr->prune(true);

				if (arr->empty())
				{
					it = map_.erase(it);
					continue;
				}
			}
			else if (auto tbl = it->second->as_table())
			{
				if (recursive)
					tbl->prune(true);

				if (tbl->empty())
				{
					it = map_.erase(it);
					continue;
				}
			}
			it++;
		}

		return *this;
	}

	TOML_EXTERNAL_LINKAGE
	void table::clear() noexcept
	{
		map_.clear();
	}

	TOML_EXTERNAL_LINKAGE
	table::map_iterator table::insert_with_hint(const_iterator hint, key && k, impl::node_ptr && v)
	{
		return map_.emplace_hint(const_map_iterator{ hint }, std::move(k), std::move(v));
	}

	TOML_EXTERNAL_LINKAGE
	bool table::equal(const table& lhs, const table& rhs) noexcept
	{
		if (&lhs == &rhs)
			return true;
		if (lhs.map_.size() != rhs.map_.size())
			return false;

		for (auto l = lhs.map_.begin(), r = rhs.map_.begin(), e = lhs.map_.end(); l != e; l++, r++)
		{
			if (l->first != r->first)
				return false;

			const auto lhs_type = l->second->type();
			const node& rhs_	= *r->second;
			const auto rhs_type = rhs_.type();
			if (lhs_type != rhs_type)
				return false;

			const bool equal = l->second->visit(
				[&](const auto& lhs_) noexcept
				{ return lhs_ == *reinterpret_cast<std::remove_reference_t<decltype(lhs_)>*>(&rhs_); });
			if (!equal)
				return false;
		}
		return true;
	}
}
TOML_NAMESPACE_END;

#include "header_end.h"
