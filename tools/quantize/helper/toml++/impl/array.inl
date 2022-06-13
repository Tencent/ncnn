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

#include "array.h"
#include "header_start.h"

TOML_NAMESPACE_START
{
	TOML_EXTERNAL_LINKAGE
	array::array() noexcept
	{
#if TOML_LIFETIME_HOOKS
		TOML_ARRAY_CREATED;
#endif
	}

	TOML_EXTERNAL_LINKAGE
	array::~array() noexcept
	{
#if TOML_LIFETIME_HOOKS
		TOML_ARRAY_DESTROYED;
#endif
	}

	TOML_EXTERNAL_LINKAGE
	array::array(const impl::array_init_elem* b, const impl::array_init_elem* e)
	{
#if TOML_LIFETIME_HOOKS
		TOML_ARRAY_CREATED;
#endif

		TOML_ASSERT_ASSUME(b);
		TOML_ASSERT_ASSUME(e);
		TOML_ASSERT_ASSUME(b <= e);

		if TOML_UNLIKELY(b == e)
			return;

		size_t cap{};
		for (auto it = b; it != e; it++)
		{
			if (it->value)
				cap++;
		}
		if TOML_UNLIKELY(!cap)
			return;

		elems_.reserve(cap);
		for (; b != e; b++)
		{
			if (b->value)
				elems_.push_back(std::move(b->value));
		}
	}

	TOML_EXTERNAL_LINKAGE
	array::array(const array& other) //
		: node(other)
	{
		elems_.reserve(other.elems_.size());
		for (const auto& elem : other)
			elems_.emplace_back(impl::make_node(elem));

#if TOML_LIFETIME_HOOKS
		TOML_ARRAY_CREATED;
#endif
	}

	TOML_EXTERNAL_LINKAGE
	array::array(array && other) noexcept //
		: node(std::move(other)),
		  elems_(std::move(other.elems_))
	{
#if TOML_LIFETIME_HOOKS
		TOML_ARRAY_CREATED;
#endif
	}

	TOML_EXTERNAL_LINKAGE
	array& array::operator=(const array& rhs)
	{
		if (&rhs != this)
		{
			node::operator=(rhs);
			elems_.clear();
			elems_.reserve(rhs.elems_.size());
			for (const auto& elem : rhs)
				elems_.emplace_back(impl::make_node(elem));
		}
		return *this;
	}

	TOML_EXTERNAL_LINKAGE
	array& array::operator=(array&& rhs) noexcept
	{
		if (&rhs != this)
		{
			node::operator=(std::move(rhs));
			elems_ = std::move(rhs.elems_);
		}
		return *this;
	}

	TOML_EXTERNAL_LINKAGE
	void array::preinsertion_resize(size_t idx, size_t count)
	{
		TOML_ASSERT(idx <= elems_.size());
		TOML_ASSERT_ASSUME(count >= 1u);
		const auto old_size			= elems_.size();
		const auto new_size			= old_size + count;
		const auto inserting_at_end = idx == old_size;
		elems_.resize(new_size);
		if (!inserting_at_end)
		{
			for (size_t left = old_size, right = new_size - 1u; left-- > idx; right--)
				elems_[right] = std::move(elems_[left]);
		}
	}

	TOML_EXTERNAL_LINKAGE
	void array::insert_at_back(impl::node_ptr && elem)
	{
		elems_.push_back(std::move(elem));
	}

	TOML_EXTERNAL_LINKAGE
	array::vector_iterator array::insert_at(const_vector_iterator pos, impl::node_ptr && elem)
	{
		return elems_.insert(pos, std::move(elem));
	}

	TOML_EXTERNAL_LINKAGE
	bool array::is_homogeneous(node_type ntype) const noexcept
	{
		if (elems_.empty())
			return false;

		if (ntype == node_type::none)
			ntype = elems_[0]->type();

		for (const auto& val : elems_)
			if (val->type() != ntype)
				return false;

		return true;
	}

	TOML_EXTERNAL_LINKAGE
	bool array::is_homogeneous(node_type ntype, node * &first_nonmatch) noexcept
	{
		if (elems_.empty())
		{
			first_nonmatch = {};
			return false;
		}
		if (ntype == node_type::none)
			ntype = elems_[0]->type();
		for (const auto& val : elems_)
		{
			if (val->type() != ntype)
			{
				first_nonmatch = val.get();
				return false;
			}
		}
		return true;
	}

	TOML_EXTERNAL_LINKAGE
	bool array::is_homogeneous(node_type ntype, const node*& first_nonmatch) const noexcept
	{
		node* fnm		  = nullptr;
		const auto result = const_cast<array&>(*this).is_homogeneous(ntype, fnm);
		first_nonmatch	  = fnm;
		return result;
	}

	TOML_EXTERNAL_LINKAGE
	node& array::at(size_t index)
	{
#if TOML_COMPILER_EXCEPTIONS

		return *elems_.at(index);

#else

		auto n = get(index);
		TOML_ASSERT_ASSUME(n && "element index not found in array!");
		return *n;

#endif
	}

	TOML_EXTERNAL_LINKAGE
	void array::reserve(size_t new_capacity)
	{
		elems_.reserve(new_capacity);
	}

	TOML_EXTERNAL_LINKAGE
	void array::shrink_to_fit()
	{
		elems_.shrink_to_fit();
	}

	TOML_EXTERNAL_LINKAGE
	void array::truncate(size_t new_size)
	{
		if (new_size < elems_.size())
			elems_.resize(new_size);
	}

	TOML_EXTERNAL_LINKAGE
	array::iterator array::erase(const_iterator pos) noexcept
	{
		return iterator{ elems_.erase(const_vector_iterator{ pos }) };
	}

	TOML_EXTERNAL_LINKAGE
	array::iterator array::erase(const_iterator first, const_iterator last) noexcept
	{
		return iterator{ elems_.erase(const_vector_iterator{ first }, const_vector_iterator{ last }) };
	}

	TOML_EXTERNAL_LINKAGE
	size_t array::total_leaf_count() const noexcept
	{
		size_t leaves{};
		for (size_t i = 0, e = elems_.size(); i < e; i++)
		{
			auto arr = elems_[i]->as_array();
			leaves += arr ? arr->total_leaf_count() : size_t{ 1 };
		}
		return leaves;
	}

	TOML_EXTERNAL_LINKAGE
	void array::flatten_child(array && child, size_t & dest_index) noexcept
	{
		for (size_t i = 0, e = child.size(); i < e; i++)
		{
			auto type = child.elems_[i]->type();
			if (type == node_type::array)
			{
				array& arr = *reinterpret_cast<array*>(child.elems_[i].get());
				if (!arr.empty())
					flatten_child(std::move(arr), dest_index);
			}
			else
				elems_[dest_index++] = std::move(child.elems_[i]);
		}
	}

	TOML_EXTERNAL_LINKAGE
	array& array::flatten()&
	{
		if (elems_.empty())
			return *this;

		bool requires_flattening	 = false;
		size_t size_after_flattening = elems_.size();
		for (size_t i = elems_.size(); i-- > 0u;)
		{
			auto arr = elems_[i]->as_array();
			if (!arr)
				continue;
			size_after_flattening--; // discount the array itself
			const auto leaf_count = arr->total_leaf_count();
			if (leaf_count > 0u)
			{
				requires_flattening = true;
				size_after_flattening += leaf_count;
			}
			else
				elems_.erase(elems_.cbegin() + static_cast<ptrdiff_t>(i));
		}

		if (!requires_flattening)
			return *this;

		elems_.reserve(size_after_flattening);

		size_t i = 0;
		while (i < elems_.size())
		{
			auto arr = elems_[i]->as_array();
			if (!arr)
			{
				i++;
				continue;
			}

			impl::node_ptr arr_storage = std::move(elems_[i]);
			const auto leaf_count	   = arr->total_leaf_count();
			if (leaf_count > 1u)
				preinsertion_resize(i + 1u, leaf_count - 1u);
			flatten_child(std::move(*arr), i); // increments i
		}

		return *this;
	}

	TOML_EXTERNAL_LINKAGE
	array& array::prune(bool recursive)& noexcept
	{
		if (elems_.empty())
			return *this;

		for (size_t i = elems_.size(); i-- > 0u;)
		{
			if (auto arr = elems_[i]->as_array())
			{
				if (recursive)
					arr->prune(true);
				if (arr->empty())
					elems_.erase(elems_.cbegin() + static_cast<ptrdiff_t>(i));
			}
			else if (auto tbl = elems_[i]->as_table())
			{
				if (recursive)
					tbl->prune(true);
				if (tbl->empty())
					elems_.erase(elems_.cbegin() + static_cast<ptrdiff_t>(i));
			}
		}

		return *this;
	}

	TOML_EXTERNAL_LINKAGE
	void array::pop_back() noexcept
	{
		elems_.pop_back();
	}

	TOML_EXTERNAL_LINKAGE
	void array::clear() noexcept
	{
		elems_.clear();
	}

	TOML_EXTERNAL_LINKAGE
	bool array::equal(const array& lhs, const array& rhs) noexcept
	{
		if (&lhs == &rhs)
			return true;
		if (lhs.elems_.size() != rhs.elems_.size())
			return false;
		for (size_t i = 0, e = lhs.elems_.size(); i < e; i++)
		{
			const auto lhs_type = lhs.elems_[i]->type();
			const node& rhs_	= *rhs.elems_[i];
			const auto rhs_type = rhs_.type();
			if (lhs_type != rhs_type)
				return false;

			const bool equal = lhs.elems_[i]->visit(
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
