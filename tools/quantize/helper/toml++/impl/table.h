//# This file is a part of toml++ and is subject to the the terms of the MIT license.
//# Copyright (c) Mark Gillard <mark.gillard@outlook.com.au>
//# See https://github.com/marzer/tomlplusplus/blob/master/LICENSE for the full license text.
// SPDX-License-Identifier: MIT
#pragma once

#include "std_map.h"
#include "std_initializer_list.h"
#include "array.h"
#include "make_node.h"
#include "node_view.h"
#include "key.h"
#include "header_start.h"

/// \cond
TOML_IMPL_NAMESPACE_START
{
	template <bool IsConst>
	struct table_proxy_pair
	{
		using value_type = std::conditional_t<IsConst, const node, node>;

		const toml::key& first;
		value_type& second;
	};

	template <bool IsConst>
	class table_iterator
	{
	  private:
		template <bool>
		friend class table_iterator;

		using proxy_type		   = table_proxy_pair<IsConst>;
		using mutable_map_iterator = std::map<toml::key, node_ptr, std::less<>>::iterator;
		using const_map_iterator   = std::map<toml::key, node_ptr, std::less<>>::const_iterator;
		using map_iterator		   = std::conditional_t<IsConst, const_map_iterator, mutable_map_iterator>;

		mutable map_iterator iter_;
		mutable std::aligned_storage_t<sizeof(proxy_type), alignof(proxy_type)> proxy_;
		mutable bool proxy_instantiated_ = false;

		TOML_NODISCARD
		proxy_type* get_proxy() const noexcept
		{
			if (!proxy_instantiated_)
			{
				auto p = ::new (static_cast<void*>(&proxy_)) proxy_type{ iter_->first, *iter_->second.get() };
				proxy_instantiated_ = true;
				return p;
			}
			else
				return TOML_LAUNDER(reinterpret_cast<proxy_type*>(&proxy_));
		}

	  public:
		TOML_NODISCARD_CTOR
		table_iterator() noexcept = default;

		TOML_NODISCARD_CTOR
		explicit table_iterator(mutable_map_iterator iter) noexcept //
			: iter_{ iter }
		{}

		TOML_CONSTRAINED_TEMPLATE(C, bool C = IsConst)
		TOML_NODISCARD_CTOR
		explicit table_iterator(const_map_iterator iter) noexcept //
			: iter_{ iter }
		{}

		TOML_CONSTRAINED_TEMPLATE(C, bool C = IsConst)
		TOML_NODISCARD_CTOR
		table_iterator(const table_iterator<false>& other) noexcept //
			: iter_{ other.iter_ }
		{}

		TOML_NODISCARD_CTOR
		table_iterator(const table_iterator& other) noexcept //
			: iter_{ other.iter_ }
		{}

		table_iterator& operator=(const table_iterator& rhs) noexcept
		{
			iter_				= rhs.iter_;
			proxy_instantiated_ = false;
			return *this;
		}

		using value_type		= table_proxy_pair<IsConst>;
		using reference			= value_type&;
		using pointer			= value_type*;
		using difference_type	= typename std::iterator_traits<map_iterator>::difference_type;
		using iterator_category = typename std::iterator_traits<map_iterator>::iterator_category;

		table_iterator& operator++() noexcept // ++pre
		{
			++iter_;
			proxy_instantiated_ = false;
			return *this;
		}

		table_iterator operator++(int) noexcept // post++
		{
			table_iterator out{ iter_ };
			++iter_;
			proxy_instantiated_ = false;
			return out;
		}

		table_iterator& operator--() noexcept // --pre
		{
			--iter_;
			proxy_instantiated_ = false;
			return *this;
		}

		table_iterator operator--(int) noexcept // post--
		{
			table_iterator out{ iter_ };
			--iter_;
			proxy_instantiated_ = false;
			return out;
		}

		TOML_PURE_INLINE_GETTER
		reference operator*() const noexcept
		{
			return *get_proxy();
		}

		TOML_PURE_INLINE_GETTER
		pointer operator->() const noexcept
		{
			return get_proxy();
		}

		TOML_PURE_INLINE_GETTER
		explicit operator const map_iterator&() const noexcept
		{
			return iter_;
		}

		TOML_CONSTRAINED_TEMPLATE(!C, bool C = IsConst)
		TOML_PURE_INLINE_GETTER
		explicit operator const const_map_iterator() const noexcept
		{
			return iter_;
		}

		TOML_PURE_INLINE_GETTER
		friend bool operator==(const table_iterator& lhs, const table_iterator& rhs) noexcept
		{
			return lhs.iter_ == rhs.iter_;
		}

		TOML_PURE_INLINE_GETTER
		friend bool operator!=(const table_iterator& lhs, const table_iterator& rhs) noexcept
		{
			return lhs.iter_ != rhs.iter_;
		}
	};

	struct table_init_pair
	{
		mutable toml::key key;
		mutable node_ptr value;

		template <typename K, typename V>
		TOML_NODISCARD_CTOR
		table_init_pair(K&& k, V&& v, value_flags flags = preserve_source_value_flags) //
			: key{ static_cast<K&&>(k) },
			  value{ make_node(static_cast<V&&>(v), flags) }
		{}
	};
}
TOML_IMPL_NAMESPACE_END;
/// \endcond

TOML_NAMESPACE_START
{
	/// \brief A BidirectionalIterator for iterating over key-value pairs in a toml::table.
	using table_iterator = POXY_IMPLEMENTATION_DETAIL(impl::table_iterator<false>);

	/// \brief A BidirectionalIterator for iterating over const key-value pairs in a toml::table.
	using const_table_iterator = POXY_IMPLEMENTATION_DETAIL(impl::table_iterator<true>);

	/// \brief	A TOML table.
	///
	/// \detail The interface of this type is modeled after std::map, with some
	/// 		additional considerations made for the heterogeneous nature of a
	/// 		TOML table.
	///
	/// \cpp
	/// toml::table tbl = toml::parse(R"(
	///
	/// 	[animals]
	/// 	cats = [ "tiger", "lion", "puma" ]
	/// 	birds = [ "macaw", "pigeon", "canary" ]
	/// 	fish = [ "salmon", "trout", "carp" ]
	///
	/// )"sv);
	///
	/// // operator[] retrieves node-views
	/// std::cout << "cats: " << tbl["animals"]["cats"] << "\n";
	/// std::cout << "fish[1]: " << tbl["animals"]["fish"][1] << "\n";
	///
	/// // at_path() does fully-qualified "toml path" lookups
	/// std::cout << "cats: " << tbl.at_path("animals.cats") << "\n";
	/// std::cout << "fish[1]: " << tbl.at_path("animals.fish[1]") << "\n";
	/// \ecpp
	///
	/// \out
	/// cats: ['tiger', 'lion', 'puma']
	/// fish[1] : 'trout'
	/// cats : ['tiger', 'lion', 'puma']
	/// fish[1] : 'trout'
	/// \eout
	class TOML_EXPORTED_CLASS table : public node
	{
	  private:
		/// \cond

		using map_type			 = std::map<toml::key, impl::node_ptr, std::less<>>;
		using map_pair			 = std::pair<const toml::key, impl::node_ptr>;
		using map_iterator		 = typename map_type::iterator;
		using const_map_iterator = typename map_type::const_iterator;
		map_type map_;

		bool inline_ = false;

		TOML_NODISCARD_CTOR
		TOML_EXPORTED_MEMBER_FUNCTION
		table(const impl::table_init_pair*, const impl::table_init_pair*);

		/// \endcond

	  public:
		/// \brief	Default constructor.
		TOML_NODISCARD_CTOR
		TOML_EXPORTED_MEMBER_FUNCTION
		table() noexcept;

		TOML_EXPORTED_MEMBER_FUNCTION
		~table() noexcept;

		/// \brief	Copy constructor.
		TOML_NODISCARD_CTOR
		TOML_EXPORTED_MEMBER_FUNCTION
		table(const table&);

		/// \brief	Move constructor.
		TOML_NODISCARD_CTOR
		TOML_EXPORTED_MEMBER_FUNCTION
		table(table&& other) noexcept;

		/// \brief	Constructs a table with one or more initial key-value pairs.
		///
		/// \detail \cpp
		/// auto tbl = toml::table{
		///		{ "foo", 1 },
		///		{ "bar", 2.0 },
		///		{ "kek", "three" }
		///	};
		/// std::cout << tbl << "\n";
		/// \ecpp
		///
		/// \out
		/// { foo = 1, bar = 2.0, kek = "three" }
		/// \eout
		///
		/// \param 	kvps	A list of key-value pairs used to initialize the table.
		TOML_NODISCARD_CTOR
		TOML_EXPORTED_MEMBER_FUNCTION
		explicit table(std::initializer_list<impl::table_init_pair> kvps) //
			: table(kvps.begin(), kvps.end())
		{}

		/// \brief	Copy-assignment operator.
		TOML_EXPORTED_MEMBER_FUNCTION
		table& operator=(const table&);

		/// \brief	Move-assignment operator.
		TOML_EXPORTED_MEMBER_FUNCTION
		table& operator=(table&& rhs) noexcept;

		/// \name Type checks
		/// @{

		/// \brief Returns #toml::node_type::table.
		TOML_CONST_INLINE_GETTER
		node_type type() const noexcept final
		{
			return node_type::table;
		}

		TOML_PURE_GETTER
		TOML_EXPORTED_MEMBER_FUNCTION
		bool is_homogeneous(node_type ntype) const noexcept final;

		TOML_PURE_GETTER
		TOML_EXPORTED_MEMBER_FUNCTION
		bool is_homogeneous(node_type ntype, node*& first_nonmatch) noexcept final;

		TOML_PURE_GETTER
		TOML_EXPORTED_MEMBER_FUNCTION
		bool is_homogeneous(node_type ntype, const node*& first_nonmatch) const noexcept final;

		/// \cond
		template <typename ElemType = void>
		TOML_PURE_GETTER
		bool is_homogeneous() const noexcept
		{
			using type = impl::unwrap_node<ElemType>;

			static_assert(
				std::is_void_v<
					type> || ((impl::is_native<type> || impl::is_one_of<type, table, array>)&&!impl::is_cvref<type>),
				"The template type argument of table::is_homogeneous() must be void or one "
				"of:" TOML_SA_UNWRAPPED_NODE_TYPE_LIST);

			return is_homogeneous(impl::node_type_of<type>);
		}
		/// \endcond

		/// \brief Returns `true`.
		TOML_CONST_INLINE_GETTER
		bool is_table() const noexcept final
		{
			return true;
		}

		/// \brief Returns `false`.
		TOML_CONST_INLINE_GETTER
		bool is_array() const noexcept final
		{
			return false;
		}

		/// \brief Returns `false`.
		TOML_PURE_GETTER
		bool is_array_of_tables() const noexcept final
		{
			return false;
		}

		/// \brief Returns `false`.
		TOML_CONST_INLINE_GETTER
		bool is_value() const noexcept final
		{
			return false;
		}

		/// \brief Returns `false`.
		TOML_CONST_INLINE_GETTER
		bool is_string() const noexcept final
		{
			return false;
		}

		/// \brief Returns `false`.
		TOML_CONST_INLINE_GETTER
		bool is_integer() const noexcept final
		{
			return false;
		}

		/// \brief Returns `false`.
		TOML_CONST_INLINE_GETTER
		bool is_floating_point() const noexcept final
		{
			return false;
		}

		/// \brief Returns `false`.
		TOML_CONST_INLINE_GETTER
		bool is_number() const noexcept final
		{
			return false;
		}

		/// \brief Returns `false`.
		TOML_CONST_INLINE_GETTER
		bool is_boolean() const noexcept final
		{
			return false;
		}

		/// \brief Returns `false`.
		TOML_CONST_INLINE_GETTER
		bool is_date() const noexcept final
		{
			return false;
		}

		/// \brief Returns `false`.
		TOML_CONST_INLINE_GETTER
		bool is_time() const noexcept final
		{
			return false;
		}

		/// \brief Returns `false`.
		TOML_CONST_INLINE_GETTER
		bool is_date_time() const noexcept final
		{
			return false;
		}

		/// @}

		/// \name Type casts
		/// @{

		/// \brief Returns a pointer to the table.
		TOML_CONST_INLINE_GETTER
		table* as_table() noexcept final
		{
			return this;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		array* as_array() noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		toml::value<std::string>* as_string() noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		toml::value<int64_t>* as_integer() noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		toml::value<double>* as_floating_point() noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		toml::value<bool>* as_boolean() noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		toml::value<date>* as_date() noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		toml::value<time>* as_time() noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		toml::value<date_time>* as_date_time() noexcept final
		{
			return nullptr;
		}

		/// \brief Returns a const-qualified pointer to the table.
		TOML_CONST_INLINE_GETTER
		const table* as_table() const noexcept final
		{
			return this;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		const array* as_array() const noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		const toml::value<std::string>* as_string() const noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		const toml::value<int64_t>* as_integer() const noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		const toml::value<double>* as_floating_point() const noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		const toml::value<bool>* as_boolean() const noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		const toml::value<date>* as_date() const noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		const toml::value<time>* as_time() const noexcept final
		{
			return nullptr;
		}

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		const toml::value<date_time>* as_date_time() const noexcept final
		{
			return nullptr;
		}

		/// @}

		/// \name Metadata
		/// @{

		/// \brief	Returns true if this table is an inline table.
		///
		/// \remarks Runtime-constructed tables (i.e. those not created during
		/// 		 parsing) are not inline by default.
		TOML_PURE_INLINE_GETTER
		bool is_inline() const noexcept
		{
			return inline_;
		}

		/// \brief	Sets whether this table is a TOML inline table.
		///
		/// \detail \godbolt{an9xdj}
		///
		/// \cpp
		/// auto tbl = toml::table{
		///		{ "a", 1 },
		///		{ "b", 2 },
		///		{ "c", 3 },
		///		{ "d", toml::table{ { "e", 4 } } }
		///	};
		/// std::cout << "is inline? "sv << tbl.is_inline() << "\n";
		/// std::cout << tbl << "\n\n";
		///
		/// tbl.is_inline(!tbl.is_inline());
		/// std::cout << "is inline? "sv << tbl.is_inline() << "\n";
		/// std::cout << tbl << "\n";
		/// \ecpp
		///
		/// \out
		/// is inline? false
		/// a = 1
		/// b = 2
		/// c = 3
		///
		/// [d]
		/// e = 4
		///
		///
		/// is inline? true
		/// { a = 1, b = 2, c = 3, d = { e = 4 } }
		/// \eout
		///
		/// \remarks A table being 'inline' is only relevent during printing;
		/// 		 it has no effect on the general functionality of the table
		/// 		 object.
		///
		/// \param 	val	The new value for 'inline'.
		void is_inline(bool val) noexcept
		{
			inline_ = val;
		}

		/// @}

		/// \name Value retrieval
		/// @{

		/// \brief	Gets the node at a specific key.
		///
		/// \detail \cpp
		/// auto tbl = toml::table{
		///		{ "a", 42, },
		///		{ "b", "is the meaning of life, apparently." }
		///	};
		///	std::cout << R"(node ["a"] exists: )"sv << !!arr.get("a") << "\n";
		///	std::cout << R"(node ["b"] exists: )"sv << !!arr.get("b") << "\n";
		///	std::cout << R"(node ["c"] exists: )"sv << !!arr.get("c") << "\n";
		/// if (auto val = arr.get("a"))
		///		std::cout << R"(node ["a"] was an )"sv << val->type() << "\n";
		/// \ecpp
		///
		/// \out
		/// node ["a"] exists: true
		/// node ["b"] exists: true
		/// node ["c"] exists: false
		/// node ["a"] was an integer
		/// \eout
		///
		/// \param 	key	The node's key.
		///
		/// \returns	A pointer to the node at the specified key, or nullptr.
		TOML_PURE_GETTER
		TOML_EXPORTED_MEMBER_FUNCTION
		node* get(std::string_view key) noexcept;

		/// \brief	Gets the node at a specific key (const overload).
		///
		/// \param 	key	The node's key.
		///
		/// \returns	A pointer to the node at the specified key, or nullptr.
		TOML_PURE_INLINE_GETTER
		const node* get(std::string_view key) const noexcept
		{
			return const_cast<table&>(*this).get(key);
		}

#if TOML_ENABLE_WINDOWS_COMPAT

		/// \brief	Gets the node at a specific key.
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \param 	key	The node's key.
		///
		/// \returns	A pointer to the node at the specified key, or nullptr.
		TOML_NODISCARD
		node* get(std::wstring_view key)
		{
			if (empty())
				return nullptr;

			return get(impl::narrow(key));
		}

		/// \brief	Gets the node at a specific key (const overload).
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \param 	key	The node's key.
		///
		/// \returns	A pointer to the node at the specified key, or nullptr.
		TOML_NODISCARD
		const node* get(std::wstring_view key) const
		{
			return const_cast<table&>(*this).get(key);
		}

#endif // TOML_ENABLE_WINDOWS_COMPAT

		/// \brief	Gets the node at a specific key if it is a particular type.
		///
		/// \detail \cpp
		/// auto tbl = toml::table{
		///		{ "a", 42, },
		///		{ "b", "is the meaning of life, apparently." }
		///	};
		/// if (auto val = arr.get_as<int64_t>("a"))
		///		std::cout << R"(node ["a"] was an integer with value )"sv << **val << "\n";
		/// \ecpp
		///
		/// \out
		/// node ["a"] was an integer with value 42
		/// \eout
		///
		/// \tparam	T		One of the TOML node or value types.
		/// \param 	key		The node's key.
		///
		/// \returns	A pointer to the node at the specified key if it was of the given type, or nullptr.
		template <typename T>
		TOML_PURE_GETTER
		impl::wrap_node<T>* get_as(std::string_view key) noexcept
		{
			const auto n = this->get(key);
			return n ? n->template as<T>() : nullptr;
		}

		/// \brief	Gets the node at a specific key if it is a particular type (const overload).
		///
		/// \tparam	T		One of the TOML node or value types.
		/// \param 	key		The node's key.
		///
		/// \returns	A pointer to the node at the specified key if it was of the given type, or nullptr.
		template <typename T>
		TOML_PURE_GETTER
		const impl::wrap_node<T>* get_as(std::string_view key) const noexcept
		{
			return const_cast<table&>(*this).template get_as<T>(key);
		}

#if TOML_ENABLE_WINDOWS_COMPAT

		/// \brief	Gets the node at a specific key if it is a particular type.
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \tparam	T		One of the TOML node or value types.
		/// \param 	key		The node's key.
		///
		/// \returns	A pointer to the node at the specified key if it was of the given type, or nullptr.
		template <typename T>
		TOML_NODISCARD
		impl::wrap_node<T>* get_as(std::wstring_view key)
		{
			if (empty())
				return nullptr;

			return get_as<T>(impl::narrow(key));
		}

		/// \brief	Gets the node at a specific key if it is a particular type (const overload).
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \tparam	T		One of the TOML node or value types.
		/// \param 	key		The node's key.
		///
		/// \returns	A pointer to the node at the specified key if it was of the given type, or nullptr.
		template <typename T>
		TOML_NODISCARD
		const impl::wrap_node<T>* get_as(std::wstring_view key) const
		{
			return const_cast<table&>(*this).template get_as<T>(key);
		}

#endif // TOML_ENABLE_WINDOWS_COMPAT

		/// \brief	Gets a reference to the element at a specific key, throwing `std::out_of_range` if none existed.
		TOML_NODISCARD
		TOML_EXPORTED_MEMBER_FUNCTION
		node& at(std::string_view key);

		/// \brief	Gets a reference to the element at a specific key, throwing `std::out_of_range` if none existed.
		TOML_NODISCARD
		const node& at(std::string_view key) const
		{
			return const_cast<table&>(*this).at(key);
		}

#if TOML_ENABLE_WINDOWS_COMPAT

		/// \brief	Gets a reference to the element at a specific key, throwing `std::out_of_range` if none existed.
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		TOML_NODISCARD
		node& at(std::wstring_view key)
		{
			return at(impl::narrow(key));
		}

		/// \brief	Gets a reference to the element at a specific key, throwing `std::out_of_range` if none existed.
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		TOML_NODISCARD
		const node& at(std::wstring_view key) const
		{
			return const_cast<table&>(*this).at(key);
		}

#endif // TOML_ENABLE_WINDOWS_COMPAT

		/// @}

		/// \name Iteration
		/// @{

		/// \brief A BidirectionalIterator for iterating over key-value pairs in a toml::table.
		using iterator = toml::table_iterator;

		/// \brief A BidirectionalIterator for iterating over const key-value pairs in a toml::table.
		using const_iterator = toml::const_table_iterator;

		/// \brief	Returns an iterator to the first key-value pair.
		TOML_PURE_INLINE_GETTER
		iterator begin() noexcept
		{
			return iterator{ map_.begin() };
		}

		/// \brief	Returns an iterator to the first key-value pair.
		TOML_PURE_INLINE_GETTER
		const_iterator begin() const noexcept
		{
			return const_iterator{ map_.cbegin() };
		}

		/// \brief	Returns an iterator to the first key-value pair.
		TOML_PURE_INLINE_GETTER
		const_iterator cbegin() const noexcept
		{
			return const_iterator{ map_.cbegin() };
		}

		/// \brief	Returns an iterator to one-past-the-last key-value pair.
		TOML_PURE_INLINE_GETTER
		iterator end() noexcept
		{
			return iterator{ map_.end() };
		}

		/// \brief	Returns an iterator to one-past-the-last key-value pair.
		TOML_PURE_INLINE_GETTER
		const_iterator end() const noexcept
		{
			return const_iterator{ map_.cend() };
		}

		/// \brief	Returns an iterator to one-past-the-last key-value pair.
		TOML_PURE_INLINE_GETTER
		const_iterator cend() const noexcept
		{
			return const_iterator{ map_.cend() };
		}

	  private:
		/// \cond

		template <typename T, typename Table>
		using for_each_value_ref = impl::copy_cvref<impl::wrap_node<impl::remove_cvref<impl::unwrap_node<T>>>, Table>;

		template <typename Func, typename Table, typename T>
		static constexpr bool can_for_each = std::is_invocable_v<Func, const key&, for_each_value_ref<T, Table>> //
										  || std::is_invocable_v<Func, for_each_value_ref<T, Table>>;

		template <typename Func, typename Table, typename T>
		static constexpr bool can_for_each_nothrow =
			std::is_nothrow_invocable_v<Func, const key&, for_each_value_ref<T, Table>> //
			|| std::is_nothrow_invocable_v<Func, for_each_value_ref<T, Table>>;

		template <typename Func, typename Table>
		static constexpr bool can_for_each_any = can_for_each<Func, Table, table>		//
											  || can_for_each<Func, Table, array>		//
											  || can_for_each<Func, Table, std::string> //
											  || can_for_each<Func, Table, int64_t>		//
											  || can_for_each<Func, Table, double>		//
											  || can_for_each<Func, Table, bool>		//
											  || can_for_each<Func, Table, date>		//
											  || can_for_each<Func, Table, time>		//
											  || can_for_each<Func, Table, date_time>;

		template <typename Func, typename Table, typename T>
		static constexpr bool for_each_is_nothrow_one = !can_for_each<Func, Table, T> //
													 || can_for_each_nothrow<Func, Table, T>;

		// clang-format off


		  template <typename Func, typename Table>
		  static constexpr bool for_each_is_nothrow = for_each_is_nothrow_one<Func, Table, table>		//
												   && for_each_is_nothrow_one<Func, Table, array>		//
												   && for_each_is_nothrow_one<Func, Table, std::string> //
												   && for_each_is_nothrow_one<Func, Table, int64_t>		//
												   && for_each_is_nothrow_one<Func, Table, double>		//
												   && for_each_is_nothrow_one<Func, Table, bool>		//
												   && for_each_is_nothrow_one<Func, Table, date>		//
												   && for_each_is_nothrow_one<Func, Table, time>		//
												   && for_each_is_nothrow_one<Func, Table, date_time>;

		// clang-format on

		template <typename Func, typename Table>
		static void do_for_each(Func&& visitor, Table&& tbl) noexcept(for_each_is_nothrow<Func&&, Table&&>)
		{
			static_assert(can_for_each_any<Func&&, Table&&>,
						  "TOML table for_each visitors must be invocable for at least one of the toml::node "
						  "specializations:" TOML_SA_NODE_TYPE_LIST);

			using kvp_type = impl::copy_cv<map_pair, std::remove_reference_t<Table>>;

			for (kvp_type& kvp : tbl.map_)
			{
				using node_ref = impl::copy_cvref<toml::node, Table&&>;
				static_assert(std::is_reference_v<node_ref>);

				const auto keep_going =
					static_cast<node_ref>(*kvp.second)
						.visit(
							[&](auto&& v)
#if !TOML_MSVC || TOML_MSVC >= 1932 // older MSVC thinks this is invalid syntax O_o
								noexcept(for_each_is_nothrow_one<Func&&, Table&&, decltype(v)>)
#endif
							{
								using value_ref = for_each_value_ref<decltype(v), Table&&>;
								static_assert(std::is_reference_v<value_ref>);

								// func(key, val)
								if constexpr (std::is_invocable_v<Func&&, const key&, value_ref>)
								{
									using return_type = decltype(static_cast<Func&&>(
										visitor)(static_cast<const key&>(kvp.first), static_cast<value_ref>(v)));

									if constexpr (impl::is_constructible_or_convertible<bool, return_type>)
									{
										return static_cast<bool>(static_cast<Func&&>(
											visitor)(static_cast<const key&>(kvp.first), static_cast<value_ref>(v)));
									}
									else
									{
										static_cast<Func&&>(visitor)(static_cast<const key&>(kvp.first),
																	 static_cast<value_ref>(v));
										return true;
									}
								}

								// func(val)
								else if constexpr (std::is_invocable_v<Func&&, value_ref>)
								{
									using return_type =
										decltype(static_cast<Func&&>(visitor)(static_cast<value_ref>(v)));

									if constexpr (impl::is_constructible_or_convertible<bool, return_type>)
									{
										return static_cast<bool>(
											static_cast<Func&&>(visitor)(static_cast<value_ref>(v)));
									}
									else
									{
										static_cast<Func&&>(visitor)(static_cast<value_ref>(v));
										return true;
									}
								}

								// visitor not compatible with this particular type
								else
									return true;
							});

				if (!keep_going)
					return;
			}
		}

		/// \endcond

	  public:
		/// \brief	Invokes a visitor on each key-value pair in the table.
		///
		/// \tparam	Func	A callable type invocable with one of the following signatures:
		///					<ul>
		///					<li> `func(key, val)`
		///					<li> `func(val)`
		///					</ul>
		///					Where:
		///					<ul>
		///					<li> `key` will recieve a const reference to a toml::key
		///					<li> `val` will recieve the value as it's concrete type with cvref-qualifications matching the table
		///					</ul>
		///					Visitors returning `bool` (or something convertible to `bool`) will cause iteration to
		///					stop if they return `false`.
		///
		/// \param 	visitor	The visitor object.
		///
		/// \returns A reference to the table.
		///
		/// \details \cpp
		/// toml::table tbl{
		///		{ "0",  0      },
		/// 	{ "1",  1      },
		/// 	{ "2",  2      },
		/// 	{ "3",  3.0    },
		/// 	{ "4",  "four" },
		/// 	{ "5",  "five" },
		/// 	{ "6",  6      }
		/// };
		///
		/// // select only the integers using a strongly-typed visitor
		/// tbl.for_each([](toml::value<int64_t>& val)
		/// {
		/// 	std::cout << val << ", ";
		/// });
		/// std::cout << "\n";
		///
		/// // select all the numeric values using a generic visitor + is_number<> metafunction
		/// tbl.for_each([](auto&& val)
		/// {
		/// 	if constexpr (toml::is_number<decltype(val)>)
		/// 		std::cout << val << ", ";
		/// });
		/// std::cout << "\n";
		///
		/// // select all the numeric values until we encounter something non-numeric
		/// tbl.for_each([](auto&& val)
		/// {
		///		if constexpr (toml::is_number<decltype(val)>)
		///		{
		///			std::cout << val << ", ";
		///			return true; // "keep going"
		///		}
		///		else
		///			return false; // "stop!"
		///
		/// });
		/// std::cout << "\n\n";
		///
		/// // visitors may also recieve the key
		/// tbl.for_each([](const toml::key& key, auto&& val)
		/// {
		///		std::cout << key << ": " << val << "\n";
		/// });
		///
		/// \ecpp
		/// \out
		/// 0, 1, 2, 6,
		/// 0, 1, 2, 3.0, 6,
		/// 0, 1, 2, 3.0,
		///
		/// 0: 0
		/// 1: 1
		/// 2: 2
		/// 3: 3.0
		/// 4: 'four'
		/// 5: 'five'
		/// 6: 6
		/// \eout
		///
		/// \see node::visit()
		template <typename Func>
		table& for_each(Func&& visitor) & noexcept(for_each_is_nothrow<Func&&, table&>)
		{
			do_for_each(static_cast<Func&&>(visitor), *this);
			return *this;
		}

		/// \brief	Invokes a visitor on each key-value pair in the table (rvalue overload).
		template <typename Func>
		table&& for_each(Func&& visitor) && noexcept(for_each_is_nothrow<Func&&, table&&>)
		{
			do_for_each(static_cast<Func&&>(visitor), static_cast<table&&>(*this));
			return static_cast<table&&>(*this);
		}

		/// \brief	Invokes a visitor on each key-value pair in the table (const lvalue overload).
		template <typename Func>
		const table& for_each(Func&& visitor) const& noexcept(for_each_is_nothrow<Func&&, const table&>)
		{
			do_for_each(static_cast<Func&&>(visitor), *this);
			return *this;
		}

		/// \brief	Invokes a visitor on each key-value pair in the table (const rvalue overload).
		template <typename Func>
		const table&& for_each(Func&& visitor) const&& noexcept(for_each_is_nothrow<Func&&, const table&&>)
		{
			do_for_each(static_cast<Func&&>(visitor), static_cast<const table&&>(*this));
			return static_cast<const table&&>(*this);
		}

		/// @}

		/// \name Size and Capacity
		/// @{

		/// \brief	Returns true if the table is empty.
		TOML_PURE_INLINE_GETTER
		bool empty() const noexcept
		{
			return map_.empty();
		}

		/// \brief	Returns the number of key-value pairs in the table.
		TOML_PURE_INLINE_GETTER
		size_t size() const noexcept
		{
			return map_.size();
		}

		/// @}

		/// \name Searching
		/// @{

	  private:
		/// \cond

		TOML_PURE_GETTER
		TOML_EXPORTED_MEMBER_FUNCTION
		map_iterator get_lower_bound(std::string_view) noexcept;

		/// \endcond

	  public:
		/// \brief Returns an iterator to the first key-value pair with key that is _not less_ than the given key.
		///
		/// \returns	An iterator to the first matching key-value pair, or #end().
		TOML_PURE_GETTER
		iterator lower_bound(std::string_view key) noexcept
		{
			return iterator{ get_lower_bound(key) };
		}

		/// \brief Returns a const iterator to the first key-value pair with key that is _not less_ than the given key.
		///
		/// \returns	An iterator to the first matching key-value pair, or #end().
		TOML_PURE_GETTER
		const_iterator lower_bound(std::string_view key) const noexcept
		{
			return const_iterator{ const_cast<table&>(*this).get_lower_bound(key) };
		}

#if TOML_ENABLE_WINDOWS_COMPAT

		/// \brief Returns an iterator to the first key-value pair with key that is _not less_ than the given key.
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \returns	An iterator to the first matching key-value pair, or #end().
		TOML_NODISCARD
		iterator lower_bound(std::wstring_view key)
		{
			if (empty())
				return end();

			return lower_bound(impl::narrow(key));
		}

		/// \brief Returns a const iterator to the first key-value pair with key that is _not less_ than the given key.
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \returns	An iterator to the first matching key-value pair, or #end().
		TOML_NODISCARD
		const_iterator lower_bound(std::wstring_view key) const
		{
			if (empty())
				return end();

			return lower_bound(impl::narrow(key));
		}

#endif // TOML_ENABLE_WINDOWS_COMPAT

		/// \brief	Gets an iterator to the node at a specific key.
		///
		/// \param 	key	The node's key.
		///
		/// \returns	An iterator to the node at the specified key, or end().
		TOML_PURE_GETTER
		TOML_EXPORTED_MEMBER_FUNCTION
		iterator find(std::string_view key) noexcept;

		/// \brief	Gets an iterator to the node at a specific key (const overload)
		///
		/// \param 	key	The node's key.
		///
		/// \returns	A const iterator to the node at the specified key, or cend().
		TOML_PURE_GETTER
		TOML_EXPORTED_MEMBER_FUNCTION
		const_iterator find(std::string_view key) const noexcept;

		/// \brief	Returns true if the table contains a node at the given key.
		TOML_PURE_GETTER
		bool contains(std::string_view key) const noexcept
		{
			return get(key) != nullptr;
		}

#if TOML_ENABLE_WINDOWS_COMPAT

		/// \brief	Gets an iterator to the node at a specific key.
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \param 	key	The node's key.
		///
		/// \returns	An iterator to the node at the specified key, or end().
		TOML_NODISCARD
		iterator find(std::wstring_view key)
		{
			if (empty())
				return end();

			return find(impl::narrow(key));
		}

		/// \brief	Gets an iterator to the node at a specific key (const overload).
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \param 	key	The node's key.
		///
		/// \returns	A const iterator to the node at the specified key, or cend().
		TOML_NODISCARD
		const_iterator find(std::wstring_view key) const
		{
			return find(impl::narrow(key));
		}

		/// \brief	Returns true if the table contains a node at the given key.
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		TOML_NODISCARD
		bool contains(std::wstring_view key) const
		{
			return contains(impl::narrow(key));
		}

#endif // TOML_ENABLE_WINDOWS_COMPAT

		/// @}

		/// \name Erasure
		/// @{

	  private:
		/// \cond

		TOML_EXPORTED_MEMBER_FUNCTION
		map_iterator erase(const_map_iterator) noexcept;

		TOML_EXPORTED_MEMBER_FUNCTION
		map_iterator erase(const_map_iterator, const_map_iterator) noexcept;

		/// \endcond

	  public:
		/// \brief	Removes the specified key-value pair from the table.
		///
		/// \detail \cpp
		/// auto tbl = toml::table{
		///		{ "a", 1 },
		///		{ "b", 2 },
		///		{ "c", 3 }
		///	};
		/// std::cout << tbl << "\n";
		///
		/// tbl.erase(tbl.begin() + 1);
		/// std::cout << tbl << "\n";
		/// \ecpp
		///
		/// \out
		/// { a = 1, b = 2, c = 3 }
		/// { a = 1, c = 3 }
		/// \eout
		///
		/// \param 	pos		Iterator to the key-value pair being erased.
		///
		/// \returns Iterator to the first key-value pair immediately following the removed key-value pair.
		iterator erase(iterator pos) noexcept
		{
			return iterator{ erase(const_map_iterator{ pos }) };
		}

		/// \brief	Removes the specified key-value pair from the table (const iterator overload).
		///
		/// \detail \cpp
		/// auto tbl = toml::table{
		///		{ "a", 1 },
		///		{ "b", 2 },
		///		{ "c", 3 }
		///	};
		/// std::cout << tbl << "\n";
		///
		/// tbl.erase(tbl.cbegin() + 1);
		/// std::cout << tbl << "\n";
		/// \ecpp
		///
		/// \out
		/// { a = 1, b = 2, c = 3 }
		/// { a = 1, c = 3 }
		/// \eout
		///
		/// \param 	pos		Iterator to the key-value pair being erased.
		///
		/// \returns Iterator to the first key-value pair immediately following the removed key-value pair.
		iterator erase(const_iterator pos) noexcept
		{
			return iterator{ erase(const_map_iterator{ pos }) };
		}

		/// \brief	Removes the key-value pairs in the range [first, last) from the table.
		///
		/// \detail \cpp
		/// auto tbl = toml::table{
		///		{ "a", 1 },
		///		{ "b", "bad" },
		///		{ "c", "karma" },
		///		{ "d", 2 }
		///	};
		/// std::cout << tbl << "\n";
		///
		/// tbl.erase(tbl.cbegin() + 1, tbl.cbegin() + 3);
		/// std::cout << tbl << "\n";
		/// \ecpp
		///
		/// \out
		/// { a = 1, b = "bad", c = "karma", d = 2 }
		/// { a = 1, d = 2 }
		/// \eout
		///
		/// \param 	begin	Iterator to the first key-value pair being erased.
		/// \param 	end		Iterator to the one-past-the-last key-value pair being erased.
		///
		/// \returns Iterator to the first key-value pair immediately following the last removed key-value pair.
		iterator erase(const_iterator begin, const_iterator end) noexcept
		{
			return iterator{ erase(const_map_iterator{ begin }, const_map_iterator{ end }) };
		}

		/// \brief	Removes the value with the given key from the table.
		///
		/// \detail \cpp
		/// auto tbl = toml::table{
		///		{ "a", 1 },
		///		{ "b", 2 },
		///		{ "c", 3 }
		///	};
		/// std::cout << tbl << "\n";
		///
		/// std::cout << tbl.erase("b") << "\n";
		/// std::cout << tbl.erase("not an existing key") << "\n";
		/// std::cout << tbl << "\n";
		/// \ecpp
		///
		/// \out
		/// { a = 1, b = 2, c = 3 }
		/// true
		/// false
		/// { a = 1, c = 3 }
		/// \eout
		///
		/// \param 	key		Key to erase.
		///
		/// \returns Number of elements removed (0 or 1).
		TOML_EXPORTED_MEMBER_FUNCTION
		size_t erase(std::string_view key) noexcept;

#if TOML_ENABLE_WINDOWS_COMPAT

		/// \brief	Removes the value with the given key from the table.
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \param 	key		Key to erase.
		///
		/// \returns Number of elements removed (0 or 1).
		size_t erase(std::wstring_view key)
		{
			if (empty())
				return false;

			return erase(impl::narrow(key));
		}

#endif // TOML_ENABLE_WINDOWS_COMPAT

		/// \brief	Removes empty child arrays and tables.
		///
		/// \detail \cpp
		///
		/// auto tbl = toml::table{ { "a", 1 }, { "b", toml::array{ } }, { "c", toml::array{ toml::table{}, toml::array{} } } };
		/// std::cout << arr << "\n";
		///
		/// arr.prune();
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// { a = 1, b = [], c = [ {}, [] ] }
		/// { a = 1 }
		/// \eout
		///
		/// \param recursive Should child arrays and tables themselves be pruned?
		///
		/// \returns A reference to the table.
		TOML_EXPORTED_MEMBER_FUNCTION
		table& prune(bool recursive = true) & noexcept;

		/// \brief	Removes empty child arrays and tables (rvalue overload).
		///
		/// \param recursive Should child arrays and tables themselves be pruned?
		///
		/// \returns An rvalue reference to the table.
		table&& prune(bool recursive = true) && noexcept
		{
			return static_cast<toml::table&&>(this->prune(recursive));
		}

		/// \brief	Removes all key-value pairs from the table.
		TOML_EXPORTED_MEMBER_FUNCTION
		void clear() noexcept;

		/// @}

		/// \name Insertion and Emplacement
		/// @{

	  private:
		/// \cond

		TOML_EXPORTED_MEMBER_FUNCTION
		map_iterator insert_with_hint(const_iterator, key&&, impl::node_ptr&&);

		/// \endcond

	  public:
		/// \brief	Emplaces a new value at a specific key if one did not already exist.
		///
		/// \tparam ValueType	toml::table, toml::array, or any native TOML value type.
		/// \tparam KeyType		A toml::key or any compatible string type.
		/// \tparam ValueArgs	Value constructor argument types.
		/// \param 	hint		Iterator to the position before which the new element will be emplaced.
		/// \param 	key			The key at which to emplace the new value.
		/// \param 	args		Arguments to forward to the value's constructor.
		///
		/// \returns An iterator to the emplacement position (or the position of the value that prevented emplacement)
		///
		/// \note This function has exactly the same semantics as [std::map::emplace_hint()](https://en.cppreference.com/w/cpp/container/map/emplace_hint).
		TOML_CONSTRAINED_TEMPLATE((is_key_or_convertible<KeyType&&> || impl::is_wide_string<KeyType>),
								  typename ValueType,
								  typename KeyType,
								  typename... ValueArgs)
		iterator emplace_hint(const_iterator hint, KeyType&& key, ValueArgs&&... args)
		{
			static_assert(!impl::is_wide_string<KeyType> || TOML_ENABLE_WINDOWS_COMPAT,
						  "Emplacement using wide-character keys is only supported on Windows with "
						  "TOML_ENABLE_WINDOWS_COMPAT enabled.");

			static_assert(!impl::is_cvref<ValueType>, "ValueType may not be const, volatile, or a reference.");

			if constexpr (impl::is_wide_string<KeyType>)
			{
#if TOML_ENABLE_WINDOWS_COMPAT
				return emplace_hint<ValueType>(hint,
											   impl::narrow(static_cast<KeyType&&>(key)),
											   static_cast<ValueArgs&&>(args)...);
#else
				static_assert(impl::dependent_false<KeyType>, "Evaluated unreachable branch!");
#endif
			}
			else
			{
				static constexpr auto moving_node_ptr = std::is_same_v<ValueType, impl::node_ptr> //
													 && sizeof...(ValueArgs) == 1u				  //
													 && impl::first_is_same<impl::node_ptr&&, ValueArgs&&...>;
				using unwrapped_type = impl::unwrap_node<ValueType>;

				static_assert(moving_node_ptr										//
								  || impl::is_native<unwrapped_type>				//
								  || impl::is_one_of<unwrapped_type, table, array>, //
							  "ValueType argument of table::emplace_hint() must be one "
							  "of:" TOML_SA_UNWRAPPED_NODE_TYPE_LIST);

				map_iterator ipos = insert_with_hint(hint, toml::key{ static_cast<KeyType&&>(key) }, nullptr);

				// if second is nullptr then we successully claimed the key and inserted the empty sentinel,
				// so now we have to construct the actual value
				if (!ipos->second)
				{
					if constexpr (moving_node_ptr)
						ipos->second = std::move(static_cast<ValueArgs&&>(args)...);
					else
					{
#if TOML_COMPILER_EXCEPTIONS
						try
						{
#endif
							ipos->second.reset(
								new impl::wrap_node<unwrapped_type>{ static_cast<ValueArgs&&>(args)... });
#if TOML_COMPILER_EXCEPTIONS
						}
						catch (...)
						{
							erase(const_map_iterator{ ipos }); // strong exception guarantee
							throw;
						}
#endif
					}
				}
				return iterator{ ipos };
			}
		}

		/// \brief	Inserts a new value at a specific key if one did not already exist.
		///
		/// \detail \godbolt{bMnW5r}
		///
		/// \cpp
		/// auto tbl = toml::table{
		///		{ "a", 1 },
		///		{ "b", 2 },
		///		{ "c", 3 }
		///	};
		/// std::cout << tbl << "\n";
		///
		/// for (auto k : { "a", "d" })
		/// {
		///		auto result = tbl.insert(k, 42);
		///		std::cout << "inserted with key '"sv << k << "': "sv << result.second << "\n";
		/// }
		/// std::cout << tbl << "\n";
		/// \ecpp
		///
		/// \out
		/// a = 1
		/// b = 2
		/// c = 3
		///
		/// inserted with key 'a': false
		/// inserted with key 'd': true
		/// a = 1
		/// b = 2
		/// c = 3
		/// d = 42
		/// \eout
		///
		/// \tparam KeyType		A toml::key or any compatible string type.
		/// \tparam ValueType	toml::node, toml::node_view, toml::table, toml::array, or a native TOML value type
		/// 					(or a type promotable to one).
		/// \param 	key			The key at which to insert the new value.
		/// \param 	val			The new value to insert.
		/// \param	flags		Value flags to apply to new values.
		///
		/// \returns	\conditional_return{Valid input}
		/// 			<ul>
		/// 				<li>An iterator to the insertion position (or the position of the value that prevented insertion)
		/// 				<li>A boolean indicating if the insertion was successful.
		/// 			</ul>
		///				\conditional_return{Input is a null toml::node_view}
		/// 			`{ end(), false }`
		///
		/// \attention The return value will always be `{ end(), false }` if the input value was an
		/// 		   null toml::node_view, because no insertion can take place. This is the only circumstance
		/// 		   in which this can occur.
		TOML_CONSTRAINED_TEMPLATE((is_key_or_convertible<KeyType&&> || impl::is_wide_string<KeyType>),
								  typename KeyType,
								  typename ValueType)
		std::pair<iterator, bool> insert(KeyType&& key,
										 ValueType&& val,
										 value_flags flags = preserve_source_value_flags)
		{
			static_assert(!impl::is_wide_string<KeyType> || TOML_ENABLE_WINDOWS_COMPAT,
						  "Insertion using wide-character keys is only supported on Windows with "
						  "TOML_ENABLE_WINDOWS_COMPAT enabled.");

			if constexpr (is_node_view<ValueType>)
			{
				if (!val)
					return { end(), false };
			}

			if constexpr (impl::is_wide_string<KeyType>)
			{
#if TOML_ENABLE_WINDOWS_COMPAT
				return insert(impl::narrow(static_cast<KeyType&&>(key)), static_cast<ValueType&&>(val), flags);
#else
				static_assert(impl::dependent_false<KeyType>, "Evaluated unreachable branch!");
#endif
			}
			else
			{
				const auto key_view = std::string_view{ key };
				map_iterator ipos	= get_lower_bound(key_view);
				if (ipos == map_.end() || ipos->first != key_view)
				{
					ipos = insert_with_hint(const_iterator{ ipos },
											toml::key{ static_cast<KeyType&&>(key) },
											impl::make_node(static_cast<ValueType&&>(val), flags));
					return { iterator{ ipos }, true };
				}
				return { iterator{ ipos }, false };
			}
		}

		/// \brief	Inserts a series of key-value pairs into the table.
		///
		/// \detail \godbolt{bzYcce}
		///
		/// \cpp
		/// auto tbl = toml::table{
		///		{ "a", 1 },
		///		{ "b", 2 },
		///		{ "c", 3 }
		///	};
		/// std::cout << tbl << "\n";
		///
		/// auto kvps = std::array<std::pair<std::string, int>, 2>{{
		///		{ "d", 42 },
		///		{ "a", 43 } // won't be inserted, 'a' already exists
		///	}};
		///	tbl.insert(kvps.begin(), kvps.end());
		/// std::cout << tbl << "\n";
		/// \ecpp
		///
		/// \out
		/// a = 1
		/// b = 2
		/// c = 3
		///
		/// a = 1
		/// b = 2
		/// c = 3
		/// d = 42
		/// \eout
		///
		/// \tparam Iter	An InputIterator to a collection of key-value pairs.
		/// \param 	begin	An iterator to the first value in the input collection.
		/// \param 	end		An iterator to one-past-the-last value in the input collection.
		/// \param	flags		Value flags to apply to new values.
		///
		/// \remarks This function is morally equivalent to calling `insert(key, value)` for each
		/// 		 key-value pair covered by the iterator range, so any values with keys already found in the
		/// 		 table will not be replaced.
		TOML_CONSTRAINED_TEMPLATE((!is_key_or_convertible<Iter> && !impl::is_wide_string<Iter>), typename Iter)
		void insert(Iter begin, Iter end, value_flags flags = preserve_source_value_flags)
		{
			if (begin == end)
				return;
			for (auto it = begin; it != end; it++)
			{
				if constexpr (std::is_rvalue_reference_v<decltype(*it)>)
					insert(std::move((*it).first), std::move((*it).second), flags);
				else
					insert((*it).first, (*it).second, flags);
			}
		}

		/// \brief	Inserts or assigns a value at a specific key.
		///
		/// \detail \godbolt{ddK563}
		///
		/// \cpp
		/// auto tbl = toml::table{
		///		{ "a", 1 },
		///		{ "b", 2 },
		///		{ "c", 3 }
		///	};
		/// std::cout << tbl << "\n";
		///
		/// for (auto k : { "a", "d" })
		/// {
		///		auto result = tbl.insert_or_assign(k, 42);
		///		std::cout << "value at key '"sv << k
		///			<< "' was "sv << (result.second ? "inserted"sv : "assigned"sv) << "\n";
		/// }
		/// std::cout << tbl << "\n";
		/// \ecpp
		///
		/// \out
		/// a = 1
		/// b = 2
		/// c = 3
		///
		/// value at key 'a' was assigned
		/// value at key 'd' was inserted
		/// a = 42
		/// b = 2
		/// c = 3
		/// d = 42
		/// \eout
		///
		/// \tparam KeyType		A toml::key or any compatible string type.
		/// \tparam ValueType	toml::node, toml::node_view, toml::table, toml::array, or a native TOML value type
		/// 					(or a type promotable to one).
		/// \param 	key			The key at which to insert or assign the value.
		/// \param 	val			The value to insert/assign.
		/// \param	flags		Value flags to apply to new values.
		///
		/// \returns	\conditional_return{Valid input}
		/// 			<ul>
		/// 				<li>An iterator to the value's position
		/// 				<li>`true` if the value was inserted, `false` if it was assigned.
		/// 			</ul>
		/// 			\conditional_return{Input is a null toml::node_view}
		/// 			 `{ end(), false }`
		///
		/// \attention The return value will always be `{ end(), false }` if the input value was
		/// 		   a null toml::node_view, because no insertion or assignment can take place.
		/// 		   This is the only circumstance in which this can occur.
		TOML_CONSTRAINED_TEMPLATE((is_key_or_convertible<KeyType&&> || impl::is_wide_string<KeyType>),
								  typename KeyType,
								  typename ValueType)
		std::pair<iterator, bool> insert_or_assign(KeyType&& key,
												   ValueType&& val,
												   value_flags flags = preserve_source_value_flags)
		{
			static_assert(!impl::is_wide_string<KeyType> || TOML_ENABLE_WINDOWS_COMPAT,
						  "Insertion using wide-character keys is only supported on Windows with "
						  "TOML_ENABLE_WINDOWS_COMPAT enabled.");

			if constexpr (is_node_view<ValueType>)
			{
				if (!val)
					return { end(), false };
			}

			if constexpr (impl::is_wide_string<KeyType>)
			{
#if TOML_ENABLE_WINDOWS_COMPAT
				return insert_or_assign(impl::narrow(static_cast<KeyType&&>(key)),
										static_cast<ValueType&&>(val),
										flags);
#else
				static_assert(impl::dependent_false<KeyType>, "Evaluated unreachable branch!");
#endif
			}
			else
			{
				const auto key_view = std::string_view{ key };
				map_iterator ipos	= get_lower_bound(key_view);
				if (ipos == map_.end() || ipos->first != key_view)
				{
					ipos = insert_with_hint(const_iterator{ ipos },
											toml::key{ static_cast<KeyType&&>(key) },
											impl::make_node(static_cast<ValueType&&>(val), flags));
					return { iterator{ ipos }, true };
				}
				else
				{
					(*ipos).second = impl::make_node(static_cast<ValueType&&>(val), flags);
					return { iterator{ ipos }, false };
				}
			}
		}

		/// \brief	Emplaces a new value at a specific key if one did not already exist.
		///
		/// \detail \cpp
		/// auto tbl = toml::table{
		///		{ "a", 1 },
		///		{ "b", 2 },
		///		{ "c", 3 }
		///	};
		/// std::cout << tbl << "\n";
		///
		/// for (auto k : { "a", "d" })
		/// {
		///		// add a string using std::string's substring constructor
		///		auto result = tbl.emplace<std::string>(k, "this is not a drill"sv, 14, 5);
		///		std::cout << "emplaced with key '"sv << k << "': "sv << result.second << "\n";
		/// }
		/// std::cout << tbl << "\n";
		/// \ecpp
		///
		/// \out
		/// { a = 1, b = 2, c = 3 }
		/// emplaced with key 'a': false
		/// emplaced with key 'd': true
		/// { a = 1, b = 2, c = 3, d = "drill" }
		/// \eout
		///
		/// \tparam ValueType	toml::table, toml::array, or any native TOML value type.
		/// \tparam KeyType		A toml::key or any compatible string type.
		/// \tparam ValueArgs	Value constructor argument types.
		/// \param 	key			The key at which to emplace the new value.
		/// \param 	args		Arguments to forward to the value's constructor.
		///
		/// \returns A std::pair containing: <br>
		/// 		- An iterator to the emplacement position (or the position of the value that prevented emplacement)
		/// 		- A boolean indicating if the emplacement was successful.
		///
		/// \remark There is no difference between insert() and emplace() for trivial value types (floats, ints, bools).
		TOML_CONSTRAINED_TEMPLATE((is_key_or_convertible<KeyType&&> || impl::is_wide_string<KeyType>),
								  typename ValueType,
								  typename KeyType,
								  typename... ValueArgs)
		std::pair<iterator, bool> emplace(KeyType&& key, ValueArgs&&... args)
		{
			static_assert(!impl::is_wide_string<KeyType> || TOML_ENABLE_WINDOWS_COMPAT,
						  "Emplacement using wide-character keys is only supported on Windows with "
						  "TOML_ENABLE_WINDOWS_COMPAT enabled.");

			static_assert(!impl::is_cvref<ValueType>, "ValueType may not be const, volatile, or a reference.");

			if constexpr (impl::is_wide_string<KeyType>)
			{
#if TOML_ENABLE_WINDOWS_COMPAT
				return emplace<ValueType>(impl::narrow(static_cast<KeyType&&>(key)), static_cast<ValueArgs&&>(args)...);
#else
				static_assert(impl::dependent_false<KeyType>, "Evaluated unreachable branch!");
#endif
			}
			else
			{
				using unwrapped_type = impl::unwrap_node<ValueType>;
				static_assert((impl::is_native<unwrapped_type> || impl::is_one_of<unwrapped_type, table, array>),
							  "ValueType argument of table::emplace() must be one "
							  "of:" TOML_SA_UNWRAPPED_NODE_TYPE_LIST);

				const auto key_view = std::string_view{ key };
				auto ipos			= get_lower_bound(key_view);
				if (ipos == map_.end() || ipos->first != key_view)
				{
					ipos = insert_with_hint(
						const_iterator{ ipos },
						toml::key{ static_cast<KeyType&&>(key) },
						impl::node_ptr{ new impl::wrap_node<unwrapped_type>{ static_cast<ValueArgs&&>(args)... } });
					return { iterator{ ipos }, true };
				}
				return { iterator{ ipos }, false };
			}
		}

		/// @}

		/// \name Node views
		/// @{

		/// \brief	Gets a node_view for the selected value.
		///
		/// \param 	key The key used for the lookup.
		///
		/// \returns	A view of the value at the given key if one existed, or an empty node view.
		///
		/// \remarks std::map::operator[]'s behaviour of default-constructing a value at a key if it
		/// 		 didn't exist is a crazy bug factory so I've deliberately chosen not to emulate it.
		/// 		 <strong>This is not an error.</strong>
		///
		/// \see toml::node_view
		TOML_NODISCARD
		node_view<node> operator[](std::string_view key) noexcept
		{
			return node_view<node>{ get(key) };
		}

		/// \brief	Gets a node_view for the selected value (const overload).
		///
		/// \param 	key The key used for the lookup.
		///
		/// \returns	A view of the value at the given key if one existed, or an empty node view.
		///
		/// \remarks std::map::operator[]'s behaviour of default-constructing a value at a key if it
		/// 		 didn't exist is a crazy bug factory so I've deliberately chosen not to emulate it.
		/// 		 <strong>This is not an error.</strong>
		///
		/// \see toml::node_view
		TOML_NODISCARD
		node_view<const node> operator[](std::string_view key) const noexcept
		{
			return node_view<const node>{ get(key) };
		}

#if TOML_ENABLE_WINDOWS_COMPAT

		/// \brief	Gets a node_view for the selected value.
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \param 	key The key used for the lookup.
		///
		/// \returns	A view of the value at the given key if one existed, or an empty node view.
		///
		/// \remarks std::map::operator[]'s behaviour of default-constructing a value at a key if it
		/// 		 didn't exist is a crazy bug factory so I've deliberately chosen not to emulate it.
		/// 		 <strong>This is not an error.</strong>
		///
		/// \see toml::node_view
		TOML_NODISCARD
		node_view<node> operator[](std::wstring_view key) noexcept
		{
			return node_view<node>{ get(key) };
		}

		/// \brief	Gets a node_view for the selected value (const overload).
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \param 	key The key used for the lookup.
		///
		/// \returns	A view of the value at the given key if one existed, or an empty node view.
		///
		/// \remarks std::map::operator[]'s behaviour of default-constructing a value at a key if it
		/// 		 didn't exist is a crazy bug factory so I've deliberately chosen not to emulate it.
		/// 		 <strong>This is not an error.</strong>
		///
		/// \see toml::node_view
		TOML_NODISCARD
		node_view<const node> operator[](std::wstring_view key) const noexcept
		{
			return node_view<const node>{ get(key) };
		}

#endif // TOML_ENABLE_WINDOWS_COMPAT

		/// @}

		/// \name Equality
		/// @{

	  private:
		/// \cond

		TOML_PURE_GETTER
		TOML_EXPORTED_STATIC_FUNCTION
		static bool equal(const table&, const table&) noexcept;

		/// \endcond
	  public:
		/// \brief	Equality operator.
		///
		/// \param 	lhs	The LHS table.
		/// \param 	rhs	The RHS table.
		///
		/// \returns	True if the tables contained the same keys and map.
		TOML_NODISCARD
		friend bool operator==(const table& lhs, const table& rhs) noexcept
		{
			return equal(lhs, rhs);
		}

		/// \brief	Inequality operator.
		///
		/// \param 	lhs	The LHS table.
		/// \param 	rhs	The RHS table.
		///
		/// \returns	True if the tables did not contain the same keys and map.
		TOML_NODISCARD
		friend bool operator!=(const table& lhs, const table& rhs) noexcept
		{
			return !equal(lhs, rhs);
		}

		/// @}

#if TOML_ENABLE_FORMATTERS

		/// \brief	Prints the table out to a stream as formatted TOML.
		///
		/// \availability This operator is only available when #TOML_ENABLE_FORMATTERS is enabled.
		friend std::ostream& operator<<(std::ostream& lhs, const table& rhs)
		{
			impl::print_to_stream(lhs, rhs);
			return lhs;
		}

#endif
	};
}
TOML_NAMESPACE_END;

#include "header_end.h"
