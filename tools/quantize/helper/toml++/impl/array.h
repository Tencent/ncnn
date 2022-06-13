//# This file is a part of toml++ and is subject to the the terms of the MIT license.
//# Copyright (c) Mark Gillard <mark.gillard@outlook.com.au>
//# See https://github.com/marzer/tomlplusplus/blob/master/LICENSE for the full license text.
// SPDX-License-Identifier: MIT
#pragma once

#include "std_utility.h"
#include "std_vector.h"
#include "std_initializer_list.h"
#include "value.h"
#include "make_node.h"
#include "header_start.h"

/// \cond
TOML_IMPL_NAMESPACE_START
{
	template <bool IsConst>
	class TOML_TRIVIAL_ABI array_iterator
	{
	  private:
		template <bool>
		friend class array_iterator;

		using mutable_vector_iterator = std::vector<node_ptr>::iterator;
		using const_vector_iterator	  = std::vector<node_ptr>::const_iterator;
		using vector_iterator		  = std::conditional_t<IsConst, const_vector_iterator, mutable_vector_iterator>;

		mutable vector_iterator iter_;

	  public:
		using value_type		= std::conditional_t<IsConst, const node, node>;
		using reference			= value_type&;
		using pointer			= value_type*;
		using difference_type	= ptrdiff_t;
		using iterator_category = typename std::iterator_traits<vector_iterator>::iterator_category;

		TOML_NODISCARD_CTOR
		array_iterator() noexcept = default;

		TOML_NODISCARD_CTOR
		explicit array_iterator(mutable_vector_iterator iter) noexcept //
			: iter_{ iter }
		{}

		TOML_CONSTRAINED_TEMPLATE(C, bool C = IsConst)
		TOML_NODISCARD_CTOR
		explicit array_iterator(const_vector_iterator iter) noexcept //
			: iter_{ iter }
		{}

		TOML_CONSTRAINED_TEMPLATE(C, bool C = IsConst)
		TOML_NODISCARD_CTOR
		array_iterator(const array_iterator<false>& other) noexcept //
			: iter_{ other.iter_ }
		{}

		TOML_NODISCARD_CTOR
		array_iterator(const array_iterator&) noexcept = default;

		array_iterator& operator=(const array_iterator&) noexcept = default;

		array_iterator& operator++() noexcept // ++pre
		{
			++iter_;
			return *this;
		}

		array_iterator operator++(int) noexcept // post++
		{
			array_iterator out{ iter_ };
			++iter_;
			return out;
		}

		array_iterator& operator--() noexcept // --pre
		{
			--iter_;
			return *this;
		}

		array_iterator operator--(int) noexcept // post--
		{
			array_iterator out{ iter_ };
			--iter_;
			return out;
		}

		TOML_PURE_INLINE_GETTER
		reference operator*() const noexcept
		{
			return *iter_->get();
		}

		TOML_PURE_INLINE_GETTER
		pointer operator->() const noexcept
		{
			return iter_->get();
		}

		TOML_PURE_INLINE_GETTER
		explicit operator const vector_iterator&() const noexcept
		{
			return iter_;
		}

		TOML_CONSTRAINED_TEMPLATE(!C, bool C = IsConst)
		TOML_PURE_INLINE_GETTER
		explicit operator const const_vector_iterator() const noexcept
		{
			return iter_;
		}

		array_iterator& operator+=(ptrdiff_t rhs) noexcept
		{
			iter_ += rhs;
			return *this;
		}

		array_iterator& operator-=(ptrdiff_t rhs) noexcept
		{
			iter_ -= rhs;
			return *this;
		}

		TOML_NODISCARD
		friend array_iterator operator+(const array_iterator& lhs, ptrdiff_t rhs) noexcept
		{
			return array_iterator{ lhs.iter_ + rhs };
		}

		TOML_NODISCARD
		friend array_iterator operator+(ptrdiff_t lhs, const array_iterator& rhs) noexcept
		{
			return array_iterator{ rhs.iter_ + lhs };
		}

		TOML_NODISCARD
		friend array_iterator operator-(const array_iterator& lhs, ptrdiff_t rhs) noexcept
		{
			return array_iterator{ lhs.iter_ - rhs };
		}

		TOML_PURE_INLINE_GETTER
		friend ptrdiff_t operator-(const array_iterator& lhs, const array_iterator& rhs) noexcept
		{
			return lhs.iter_ - rhs.iter_;
		}

		TOML_PURE_INLINE_GETTER
		friend bool operator==(const array_iterator& lhs, const array_iterator& rhs) noexcept
		{
			return lhs.iter_ == rhs.iter_;
		}

		TOML_PURE_INLINE_GETTER
		friend bool operator!=(const array_iterator& lhs, const array_iterator& rhs) noexcept
		{
			return lhs.iter_ != rhs.iter_;
		}

		TOML_PURE_INLINE_GETTER
		friend bool operator<(const array_iterator& lhs, const array_iterator& rhs) noexcept
		{
			return lhs.iter_ < rhs.iter_;
		}

		TOML_PURE_INLINE_GETTER
		friend bool operator<=(const array_iterator& lhs, const array_iterator& rhs) noexcept
		{
			return lhs.iter_ <= rhs.iter_;
		}

		TOML_PURE_INLINE_GETTER
		friend bool operator>(const array_iterator& lhs, const array_iterator& rhs) noexcept
		{
			return lhs.iter_ > rhs.iter_;
		}

		TOML_PURE_INLINE_GETTER
		friend bool operator>=(const array_iterator& lhs, const array_iterator& rhs) noexcept
		{
			return lhs.iter_ >= rhs.iter_;
		}

		TOML_PURE_INLINE_GETTER
		reference operator[](ptrdiff_t idx) const noexcept
		{
			return *(iter_ + idx)->get();
		}
	};

	struct array_init_elem
	{
		mutable node_ptr value;

		template <typename T>
		TOML_NODISCARD_CTOR
		array_init_elem(T&& val, value_flags flags = preserve_source_value_flags) //
			: value{ make_node(static_cast<T&&>(val), flags) }
		{}
	};
}
TOML_IMPL_NAMESPACE_END;
/// \endcond

TOML_NAMESPACE_START
{
	/// \brief A RandomAccessIterator for iterating over elements in a toml::array.
	using array_iterator = POXY_IMPLEMENTATION_DETAIL(impl::array_iterator<false>);

	/// \brief A RandomAccessIterator for iterating over const elements in a toml::array.
	using const_array_iterator = POXY_IMPLEMENTATION_DETAIL(impl::array_iterator<true>);

	/// \brief	A TOML array.
	///
	/// \detail The interface of this type is modeled after std::vector, with some
	/// 		additional considerations made for the heterogeneous nature of a
	/// 		TOML array.
	///
	/// \godbolt{sjK4da}
	///
	/// \cpp
	///
	/// toml::table tbl = toml::parse(R"(
	///     arr = [1, 2, 3, 4, 'five']
	/// )"sv);
	///
	/// // get the element as an array
	/// toml::array& arr = *tbl.get_as<toml::array>("arr");
	/// std::cout << arr << "\n";
	///
	/// // increment each element with visit()
	/// for (auto&& elem : arr)
	/// {
	/// 	elem.visit([](auto&& el) noexcept
	/// 	{
	/// 		if constexpr (toml::is_number<decltype(el)>)
	/// 			(*el)++;
	/// 		else if constexpr (toml::is_string<decltype(el)>)
	/// 			el = "six"sv;
	/// 	});
	/// }
	/// std::cout << arr << "\n";
	///
	/// // add and remove elements
	/// arr.push_back(7);
	/// arr.push_back(8.0f);
	/// arr.push_back("nine"sv);
	/// arr.erase(arr.cbegin());
	/// std::cout << arr << "\n";
	///
	/// // emplace elements
	/// arr.emplace_back<std::string>("ten");
	/// arr.emplace_back<toml::array>(11, 12.0);
	/// std::cout << arr << "\n";
	/// \ecpp
	///
	/// \out
	/// [ 1, 2, 3, 4, 'five' ]
	/// [ 2, 3, 4, 5, 'six' ]
	/// [ 3, 4, 5, 'six', 7, 8.0, 'nine' ]
	/// [ 3, 4, 5, 'six', 7, 8.0, 'nine', 'ten', [ 11, 12.0 ] ]
	/// \eout
	class TOML_EXPORTED_CLASS array : public node
	{
	  private:
		/// \cond

		using vector_type			= std::vector<impl::node_ptr>;
		using vector_iterator		= typename vector_type::iterator;
		using const_vector_iterator = typename vector_type::const_iterator;
		vector_type elems_;

		TOML_NODISCARD_CTOR
		TOML_EXPORTED_MEMBER_FUNCTION
		array(const impl::array_init_elem*, const impl::array_init_elem*);

		TOML_NODISCARD_CTOR
		array(std::false_type, std::initializer_list<impl::array_init_elem> elems) //
			: array{ elems.begin(), elems.end() }
		{}

		TOML_EXPORTED_MEMBER_FUNCTION
		void preinsertion_resize(size_t idx, size_t count);

		TOML_EXPORTED_MEMBER_FUNCTION
		void insert_at_back(impl::node_ptr&&);

		TOML_EXPORTED_MEMBER_FUNCTION
		vector_iterator insert_at(const_vector_iterator, impl::node_ptr&&);

		template <typename T>
		void emplace_back_if_not_empty_view(T&& val, value_flags flags)
		{
			if constexpr (is_node_view<T>)
			{
				if (!val)
					return;
			}
			insert_at_back(impl::make_node(static_cast<T&&>(val), flags));
		}

		TOML_NODISCARD
		TOML_EXPORTED_MEMBER_FUNCTION
		size_t total_leaf_count() const noexcept;

		TOML_EXPORTED_MEMBER_FUNCTION
		void flatten_child(array&& child, size_t& dest_index) noexcept;

		/// \endcond

	  public:
		using value_type	  = node;
		using size_type		  = size_t;
		using difference_type = ptrdiff_t;
		using reference		  = node&;
		using const_reference = const node&;

		/// \brief	Default constructor.
		TOML_NODISCARD_CTOR
		TOML_EXPORTED_MEMBER_FUNCTION
		array() noexcept;

		TOML_EXPORTED_MEMBER_FUNCTION
		~array() noexcept;

		/// \brief	Copy constructor.
		TOML_NODISCARD_CTOR
		TOML_EXPORTED_MEMBER_FUNCTION
		array(const array&);

		/// \brief	Move constructor.
		TOML_NODISCARD_CTOR
		TOML_EXPORTED_MEMBER_FUNCTION
		array(array&& other) noexcept;

		/// \brief	Constructs an array with one or more initial elements.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 1, 2.0, "three"sv, toml::array{ 4, 5 } };
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 2.0, 'three', [ 4, 5 ] ]
		/// \eout
		///
		/// \remark	\parblock If you need to construct an array with one child array element, the array's move constructor
		/// 		will take precedence and perform a move-construction instead. You can use toml::inserter to
		/// 		suppress this behaviour: \cpp
		/// // desired result: [ [ 42 ] ]
		/// auto bad = toml::array{ toml::array{ 42 } }
		/// auto good = toml::array{ toml::inserter{ toml::array{ 42 } } }
		/// std::cout << "bad: " << bad << "\n";
		/// std::cout << "good:" << good << "\n";
		/// \ecpp
		///
		/// \out
		/// bad:  [ 42 ]
		/// good: [ [ 42 ] ]
		/// \eout
		///
		/// \endparblock
		///
		/// \tparam	ElemType	One of the TOML node or value types (or a type promotable to one).
		/// \tparam	ElemTypes	One of the TOML node or value types (or a type promotable to one).
		/// \param 	val 	The node or value used to initialize element 0.
		/// \param 	vals	The nodes or values used to initialize elements 1...N.
		TOML_CONSTRAINED_TEMPLATE((sizeof...(ElemTypes) > 0 || !std::is_same_v<impl::remove_cvref<ElemType>, array>),
								  typename ElemType,
								  typename... ElemTypes)
		TOML_NODISCARD_CTOR
		explicit array(ElemType&& val, ElemTypes&&... vals)
			: array{ std::false_type{},
					 std::initializer_list<impl::array_init_elem>{ static_cast<ElemType&&>(val),
																   static_cast<ElemTypes&&>(vals)... } }
		{}

		/// \brief	Copy-assignment operator.
		TOML_EXPORTED_MEMBER_FUNCTION
		array& operator=(const array&);

		/// \brief	Move-assignment operator.
		TOML_EXPORTED_MEMBER_FUNCTION
		array& operator=(array&& rhs) noexcept;

		/// \name Type checks
		/// @{

		/// \brief Returns #toml::node_type::array.
		TOML_CONST_INLINE_GETTER
		node_type type() const noexcept final
		{
			return node_type::array;
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
			using unwrapped_type = impl::unwrap_node<impl::remove_cvref<ElemType>>;
			static_assert(std::is_void_v<unwrapped_type> //
							  || (impl::is_native<unwrapped_type> || impl::is_one_of<unwrapped_type, table, array>),
						  "The template type argument of array::is_homogeneous() must be void or one "
						  "of:" TOML_SA_UNWRAPPED_NODE_TYPE_LIST);

			return is_homogeneous(impl::node_type_of<unwrapped_type>);
		}
		/// \endcond

		/// \brief Returns `false`.
		TOML_CONST_INLINE_GETTER
		bool is_table() const noexcept final
		{
			return false;
		}

		/// \brief Returns `true`.
		TOML_CONST_INLINE_GETTER
		bool is_array() const noexcept final
		{
			return true;
		}

		/// \brief Returns `true` if the array contains only tables.
		TOML_PURE_GETTER
		bool is_array_of_tables() const noexcept final
		{
			return is_homogeneous(node_type::table);
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

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		table* as_table() noexcept final
		{
			return nullptr;
		}

		/// \brief Returns a pointer to the array.
		TOML_CONST_INLINE_GETTER
		array* as_array() noexcept final
		{
			return this;
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

		/// \brief Returns `nullptr`.
		TOML_CONST_INLINE_GETTER
		const table* as_table() const noexcept final
		{
			return nullptr;
		}

		/// \brief Returns a const-qualified pointer to the array.
		TOML_CONST_INLINE_GETTER
		const array* as_array() const noexcept final
		{
			return this;
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

		/// \name Value retrieval
		/// @{

		/// \brief	Gets a pointer to the element at a specific index.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 99, "bottles of beer on the wall" };
		///	std::cout << "element [0] exists: "sv << !!arr.get(0) << "\n";
		///	std::cout << "element [1] exists: "sv << !!arr.get(1) << "\n";
		///	std::cout << "element [2] exists: "sv << !!arr.get(2) << "\n";
		/// if (toml::node* val = arr.get(0))
		///		std::cout << "element [0] is an "sv << val->type() << "\n";
		/// \ecpp
		///
		/// \out
		/// element [0] exists: true
		/// element [1] exists: true
		/// element [2] exists: false
		/// element [0] is an integer
		/// \eout
		///
		/// \param 	index	The element's index.
		///
		/// \returns	A pointer to the element at the specified index if one existed, or nullptr.
		TOML_PURE_INLINE_GETTER
		node* get(size_t index) noexcept
		{
			return index < elems_.size() ? elems_[index].get() : nullptr;
		}

		/// \brief	Gets a pointer to the element at a specific index (const overload).
		///
		/// \param 	index	The element's index.
		///
		/// \returns	A pointer to the element at the specified index if one existed, or nullptr.
		TOML_PURE_INLINE_GETTER
		const node* get(size_t index) const noexcept
		{
			return const_cast<array&>(*this).get(index);
		}

		/// \brief	Gets a pointer to the element at a specific index if it is a particular type.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 42, "is the meaning of life, apparently."sv };
		/// if (toml::value<int64_t>* val = arr.get_as<int64_t>(0))
		///		std::cout << "element [0] is an integer with value "sv << *val << "\n";
		/// \ecpp
		///
		/// \out
		/// element [0] is an integer with value 42
		/// \eout
		///
		/// \tparam ElemType	toml::table, toml::array, or a native TOML value type
		/// \param 	index		The element's index.
		///
		/// \returns	A pointer to the selected element if it existed and was of the specified type, or nullptr.
		template <typename ElemType>
		TOML_NODISCARD
		impl::wrap_node<ElemType>* get_as(size_t index) noexcept
		{
			if (auto val = get(index))
				return val->template as<ElemType>();
			return nullptr;
		}

		/// \brief	Gets a pointer to the element at a specific index if it is a particular type (const overload).
		///
		/// \tparam ElemType	toml::table, toml::array, or a native TOML value type
		/// \param 	index		The element's index.
		///
		/// \returns	A pointer to the selected element if it existed and was of the specified type, or nullptr.
		template <typename ElemType>
		TOML_NODISCARD
		const impl::wrap_node<ElemType>* get_as(size_t index) const noexcept
		{
			return const_cast<array&>(*this).template get_as<ElemType>(index);
		}

		/// \brief	Gets a reference to the element at a specific index.
		TOML_NODISCARD
		node& operator[](size_t index) noexcept
		{
			return *elems_[index];
		}

		/// \brief	Gets a reference to the element at a specific index.
		TOML_NODISCARD
		const node& operator[](size_t index) const noexcept
		{
			return *elems_[index];
		}

		/// \brief	Gets a reference to the element at a specific index, throwing `std::out_of_range` if none existed.
		TOML_NODISCARD
		TOML_EXPORTED_MEMBER_FUNCTION
		node& at(size_t index);

		/// \brief	Gets a reference to the element at a specific index, throwing `std::out_of_range` if none existed.
		TOML_NODISCARD
		const node& at(size_t index) const
		{
			return const_cast<array&>(*this).at(index);
		}

		/// \brief	Returns a reference to the first element in the array.
		TOML_NODISCARD
		node& front() noexcept
		{
			return *elems_.front();
		}

		/// \brief	Returns a reference to the first element in the array.
		TOML_NODISCARD
		const node& front() const noexcept
		{
			return *elems_.front();
		}

		/// \brief	Returns a reference to the last element in the array.
		TOML_NODISCARD
		node& back() noexcept
		{
			return *elems_.back();
		}

		/// \brief	Returns a reference to the last element in the array.
		TOML_NODISCARD
		const node& back() const noexcept
		{
			return *elems_.back();
		}

		/// @}

		/// \name Iteration
		/// @{

		/// \brief A RandomAccessIterator for iterating over elements in a toml::array.
		using iterator = array_iterator;

		/// \brief A RandomAccessIterator for iterating over const elements in a toml::array.
		using const_iterator = const_array_iterator;

		/// \brief	Returns an iterator to the first element.
		TOML_NODISCARD
		iterator begin() noexcept
		{
			return iterator{ elems_.begin() };
		}

		/// \brief	Returns an iterator to the first element.
		TOML_NODISCARD
		const_iterator begin() const noexcept
		{
			return const_iterator{ elems_.cbegin() };
		}

		/// \brief	Returns an iterator to the first element.
		TOML_NODISCARD
		const_iterator cbegin() const noexcept
		{
			return const_iterator{ elems_.cbegin() };
		}

		/// \brief	Returns an iterator to one-past-the-last element.
		TOML_NODISCARD
		iterator end() noexcept
		{
			return iterator{ elems_.end() };
		}

		/// \brief	Returns an iterator to one-past-the-last element.
		TOML_NODISCARD
		const_iterator end() const noexcept
		{
			return const_iterator{ elems_.cend() };
		}

		/// \brief	Returns an iterator to one-past-the-last element.
		TOML_NODISCARD
		const_iterator cend() const noexcept
		{
			return const_iterator{ elems_.cend() };
		}

	  private:
		/// \cond

		template <typename T, typename Array>
		using for_each_elem_ref = impl::copy_cvref<impl::wrap_node<impl::remove_cvref<impl::unwrap_node<T>>>, Array>;

		template <typename Func, typename Array, typename T>
		static constexpr bool can_for_each = std::is_invocable_v<Func, for_each_elem_ref<T, Array>, size_t> //
										  || std::is_invocable_v<Func, size_t, for_each_elem_ref<T, Array>> //
										  || std::is_invocable_v<Func, for_each_elem_ref<T, Array>>;

		template <typename Func, typename Array, typename T>
		static constexpr bool can_for_each_nothrow =
			std::is_nothrow_invocable_v<Func, for_each_elem_ref<T, Array>, size_t>	  //
			|| std::is_nothrow_invocable_v<Func, size_t, for_each_elem_ref<T, Array>> //
			|| std::is_nothrow_invocable_v<Func, for_each_elem_ref<T, Array>>;

		template <typename Func, typename Array>
		static constexpr bool can_for_each_any = can_for_each<Func, Array, table>		//
											  || can_for_each<Func, Array, array>		//
											  || can_for_each<Func, Array, std::string> //
											  || can_for_each<Func, Array, int64_t>		//
											  || can_for_each<Func, Array, double>		//
											  || can_for_each<Func, Array, bool>		//
											  || can_for_each<Func, Array, date>		//
											  || can_for_each<Func, Array, time>		//
											  || can_for_each<Func, Array, date_time>;

		template <typename Func, typename Array, typename T>
		static constexpr bool for_each_is_nothrow_one = !can_for_each<Func, Array, T> //
													 || can_for_each_nothrow<Func, Array, T>;

		// clang-format off


		template <typename Func, typename Array>
		static constexpr bool for_each_is_nothrow = for_each_is_nothrow_one<Func, Array, table>		  //
												 && for_each_is_nothrow_one<Func, Array, array>		  //
												 && for_each_is_nothrow_one<Func, Array, std::string> //
												 && for_each_is_nothrow_one<Func, Array, int64_t>	  //
												 && for_each_is_nothrow_one<Func, Array, double>	  //
												 && for_each_is_nothrow_one<Func, Array, bool>		  //
												 && for_each_is_nothrow_one<Func, Array, date>		  //
												 && for_each_is_nothrow_one<Func, Array, time>		  //
												 && for_each_is_nothrow_one<Func, Array, date_time>;

		// clang-format on

		template <typename Func, typename Array>
		static void do_for_each(Func&& visitor, Array&& arr) noexcept(for_each_is_nothrow<Func&&, Array&&>)
		{
			static_assert(can_for_each_any<Func&&, Array&&>,
						  "TOML array for_each visitors must be invocable for at least one of the toml::node "
						  "specializations:" TOML_SA_NODE_TYPE_LIST);

			for (size_t i = 0; i < arr.size(); i++)
			{
				using node_ref = impl::copy_cvref<toml::node, Array&&>;
				static_assert(std::is_reference_v<node_ref>);

				const auto keep_going =
					static_cast<node_ref>(static_cast<Array&&>(arr)[i])
						.visit(
							[&](auto&& elem)
#if !TOML_MSVC || TOML_MSVC >= 1932 // older MSVC thinks this is invalid syntax O_o
								noexcept(for_each_is_nothrow_one<Func&&, Array&&, decltype(elem)>)
#endif
							{
								using elem_ref = for_each_elem_ref<decltype(elem), Array&&>;
								static_assert(std::is_reference_v<elem_ref>);

								// func(elem, i)
								if constexpr (std::is_invocable_v<Func&&, elem_ref, size_t>)
								{
									using return_type =
										decltype(static_cast<Func&&>(visitor)(static_cast<elem_ref>(elem), i));

									if constexpr (impl::is_constructible_or_convertible<bool, return_type>)
									{
										return static_cast<bool>(
											static_cast<Func&&>(visitor)(static_cast<elem_ref>(elem), i));
									}
									else
									{
										static_cast<Func&&>(visitor)(static_cast<elem_ref>(elem), i);
										return true;
									}
								}

								// func(i, elem)
								else if constexpr (std::is_invocable_v<Func&&, size_t, elem_ref>)
								{
									using return_type =
										decltype(static_cast<Func&&>(visitor)(i, static_cast<elem_ref>(elem)));

									if constexpr (impl::is_constructible_or_convertible<bool, return_type>)
									{
										return static_cast<bool>(
											static_cast<Func&&>(visitor)(i, static_cast<elem_ref>(elem)));
									}
									else
									{
										static_cast<Func&&>(visitor)(i, static_cast<elem_ref>(elem));
										return true;
									}
								}

								// func(elem)
								else if constexpr (std::is_invocable_v<Func&&, elem_ref>)
								{
									using return_type =
										decltype(static_cast<Func&&>(visitor)(static_cast<elem_ref>(elem)));

									if constexpr (impl::is_constructible_or_convertible<bool, return_type>)
									{
										return static_cast<bool>(
											static_cast<Func&&>(visitor)(static_cast<elem_ref>(elem)));
									}
									else
									{
										static_cast<Func&&>(visitor)(static_cast<elem_ref>(elem));
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
		/// \brief	Invokes a visitor on each element in the array.
		///
		/// \tparam	Func	A callable type invocable with one of the following signatures:
		///					<ul>
		///					<li> `func(elem, index)`
		///					<li> `func(elem)`
		///					<li> `func(index, elem)`
		///					</ul>
		///					Where:
		///					<ul>
		///					<li> `elem` will recieve the element as it's concrete type with cvref-qualifications matching the array
		///					<li> `index` will recieve a `size_t` indicating the element's index
		///					</ul>
		///					Visitors returning `bool` (or something convertible to `bool`) will cause iteration to
		///					stop if they return `false`.
		///
		/// \param 	visitor	The visitor object.
		///
		/// \returns A reference to the array.
		///
		/// \details \cpp
		/// toml::array arr{ 0, 1, 2, 3.0, "four", "five", 6 };
		///
		/// // select only the integers using a strongly-typed visitor
		/// arr.for_each([](toml::value<int64_t>& elem)
		/// {
		///		std::cout << elem << ", ";
		/// });
		/// std::cout << "\n";
		///
		/// // select all the numeric values using a generic visitor + is_number<> metafunction
		/// arr.for_each([](auto&& elem)
		/// {
		///		if constexpr (toml::is_number<decltype(elem)>)
		///			std::cout << elem << ", ";
		/// });
		/// std::cout << "\n";
		///
		/// // select all the numeric values until we encounter something non-numeric
		/// arr.for_each([](auto&& elem)
		/// {
		///		if constexpr (toml::is_number<decltype(elem)>)
		///		{
		///			std::cout << elem << ", ";
		///			return true; // "keep going"
		///		}
		///		else
		///			return false; // "stop!"
		///
		/// });
		/// std::cout << "\n";
		///
		/// \ecpp
		/// \out
		/// 0, 1, 2, 6,
		/// 0, 1, 2, 3.0, 6,
		/// 0, 1, 2, 3.0,
		/// \eout
		///
		/// \see node::visit()
		template <typename Func>
		array& for_each(Func&& visitor) & noexcept(for_each_is_nothrow<Func&&, array&>)
		{
			do_for_each(static_cast<Func&&>(visitor), *this);
			return *this;
		}

		/// \brief	Invokes a visitor on each element in the array (rvalue overload).
		template <typename Func>
		array&& for_each(Func&& visitor) && noexcept(for_each_is_nothrow<Func&&, array&&>)
		{
			do_for_each(static_cast<Func&&>(visitor), static_cast<array&&>(*this));
			return static_cast<array&&>(*this);
		}

		/// \brief	Invokes a visitor on each element in the array (const lvalue overload).
		template <typename Func>
		const array& for_each(Func&& visitor) const& noexcept(for_each_is_nothrow<Func&&, const array&>)
		{
			do_for_each(static_cast<Func&&>(visitor), *this);
			return *this;
		}

		/// \brief	Invokes a visitor on each element in the array (const rvalue overload).
		template <typename Func>
		const array&& for_each(Func&& visitor) const&& noexcept(for_each_is_nothrow<Func&&, const array&&>)
		{
			do_for_each(static_cast<Func&&>(visitor), static_cast<const array&&>(*this));
			return static_cast<const array&&>(*this);
		}

		/// @}

		/// \name Size and Capacity
		/// @{

		/// \brief	Returns true if the array is empty.
		TOML_NODISCARD
		bool empty() const noexcept
		{
			return elems_.empty();
		}

		/// \brief	Returns the number of elements in the array.
		TOML_NODISCARD
		size_t size() const noexcept
		{
			return elems_.size();
		}

		/// \brief	Returns the maximum number of elements that can be stored in an array on the current platform.
		TOML_NODISCARD
		size_t max_size() const noexcept
		{
			return elems_.max_size();
		}

		/// \brief	Returns the current max number of elements that may be held in the array's internal storage.
		TOML_NODISCARD
		size_t capacity() const noexcept
		{
			return elems_.capacity();
		}

		/// \brief	Reserves internal storage capacity up to a pre-determined number of elements.
		TOML_EXPORTED_MEMBER_FUNCTION
		void reserve(size_t new_capacity);

		/// \brief	Requests the removal of any unused internal storage capacity.
		TOML_EXPORTED_MEMBER_FUNCTION
		void shrink_to_fit();

		/// \brief	Shrinks the array to the given size.
		///
		/// \detail \godbolt{rxEzK5}
		///
		/// \cpp
		/// auto arr = toml::array{ 1, 2, 3 };
		/// std::cout << arr << "\n";
		///
		/// arr.truncate(5); // no-op
		/// std::cout << arr << "\n";
		///
		/// arr.truncate(1);
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 2, 3 ]
		/// [ 1, 2, 3 ]
		/// [ 1]
		/// \eout
		///
		/// \remarks	Does nothing if the requested size is larger than or equal to the current size.
		TOML_EXPORTED_MEMBER_FUNCTION
		void truncate(size_t new_size);

		/// \brief	Resizes the array.
		///
		/// \detail \godbolt{W5zqx3}
		///
		/// \cpp
		/// auto arr = toml::array{ 1, 2, 3 };
		/// std::cout << arr << "\n";
		///
		/// arr.resize(6, 42);
		/// std::cout << arr << "\n";
		///
		/// arr.resize(2, 0);
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 2, 3 ]
		/// [ 1, 2, 3, 42, 42, 42 ]
		/// [ 1, 2 ]
		/// \eout
		///
		/// \tparam ElemType	toml::node, toml::table, toml::array, or a native TOML value type
		/// 					(or a type promotable to one).
		///
		/// \param 	new_size			The number of elements the array will have after resizing.
		/// \param 	default_init_val	The node or value used to initialize new elements if the array needs to grow.
		/// \param	default_init_flags	Value flags to apply to new values created if new elements are created by growing.
		template <typename ElemType>
		void resize(size_t new_size,
					ElemType&& default_init_val,
					value_flags default_init_flags = preserve_source_value_flags)
		{
			static_assert(!is_node_view<ElemType>,
						  "The default element type argument to toml::array::resize may not be toml::node_view.");

			if (!new_size)
				clear();
			else if (new_size > elems_.size())
				insert(cend(), new_size - elems_.size(), static_cast<ElemType&&>(default_init_val), default_init_flags);
			else
				truncate(new_size);
		}

		/// @}

		/// \name Erasure
		/// @{

		/// \brief	Removes the specified element from the array.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 1, 2, 3 };
		/// std::cout << arr << "\n";
		///
		/// arr.erase(arr.cbegin() + 1);
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 2, 3 ]
		/// [ 1, 3 ]
		/// \eout
		///
		/// \param 	pos		Iterator to the element being erased.
		///
		/// \returns Iterator to the first element immediately following the removed element.
		TOML_EXPORTED_MEMBER_FUNCTION
		iterator erase(const_iterator pos) noexcept;

		/// \brief	Removes the elements in the range [first, last) from the array.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 1, "bad", "karma" 2 };
		/// std::cout << arr << "\n";
		///
		/// arr.erase(arr.cbegin() + 1, arr.cbegin() + 3);
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 'bad', 'karma', 3 ]
		/// [ 1, 3 ]
		/// \eout
		///
		/// \param 	first	Iterator to the first element being erased.
		/// \param 	last	Iterator to the one-past-the-last element being erased.
		///
		/// \returns Iterator to the first element immediately following the last removed element.
		TOML_EXPORTED_MEMBER_FUNCTION
		iterator erase(const_iterator first, const_iterator last) noexcept;

		/// \brief	Flattens this array, recursively hoisting the contents of child arrays up into itself.
		///
		/// \detail \cpp
		///
		/// auto arr = toml::array{ 1, 2, toml::array{ 3, 4, toml::array{ 5 } }, 6, toml::array{} };
		/// std::cout << arr << "\n";
		///
		/// arr.flatten();
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 2, [ 3, 4, [ 5 ] ], 6, [] ]
		/// [ 1, 2, 3, 4, 5, 6 ]
		/// \eout
		///
		/// \remarks	Arrays inside child tables are not flattened.
		///
		/// \returns A reference to the array.
		TOML_EXPORTED_MEMBER_FUNCTION
		array& flatten() &;

		/// \brief	 Flattens this array, recursively hoisting the contents of child arrays up into itself (rvalue overload).
		array&& flatten() &&
		{
			return static_cast<toml::array&&>(this->flatten());
		}

		/// \brief	Removes empty child arrays and tables.
		///
		/// \detail \cpp
		///
		/// auto arr = toml::array{ 1, 2, toml::array{ }, toml::array{ 3, toml::array{ } }, 4 };
		/// std::cout << arr << "\n";
		///
		/// arr.prune(true);
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 2, [], [ 3, [] ], 4 ]
		/// [ 1, 2, [ 3 ], 4 ]
		/// \eout
		///
		/// \param recursive Should child arrays and tables themselves be pruned?
		///
		/// \returns A reference to the array.
		TOML_EXPORTED_MEMBER_FUNCTION
		array& prune(bool recursive = true) & noexcept;

		/// \brief	Removes empty child arrays and tables (rvalue overload).
		///
		/// \param recursive Should child arrays and tables themselves be pruned?
		///
		/// \returns An rvalue reference to the array.
		array&& prune(bool recursive = true) && noexcept
		{
			return static_cast<toml::array&&>(this->prune(recursive));
		}

		/// \brief	Removes the last element from the array.
		TOML_EXPORTED_MEMBER_FUNCTION
		void pop_back() noexcept;

		/// \brief	Removes all elements from the array.
		TOML_EXPORTED_MEMBER_FUNCTION
		void clear() noexcept;

		/// @}

		/// \name Insertion and Emplacement
		/// @{

		/// \brief	Inserts a new element at a specific position in the array.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 1, 3 };
		///	arr.insert(arr.cbegin() + 1, "two");
		///	arr.insert(arr.cend(), toml::array{ 4, 5 });
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 'two', 3, [ 4, 5 ] ]
		/// \eout
		///
		/// \tparam ElemType	toml::node, toml::node_view, toml::table, toml::array, or a native TOML value type
		/// 					(or a type promotable to one).
		/// \param 	pos			The insertion position.
		/// \param 	val			The node or value being inserted.
		/// \param	flags		Value flags to apply to new values.
		///
		/// \returns \conditional_return{Valid input}
		///			 An iterator to the newly-inserted element.
		///			 \conditional_return{Input is a null toml::node_view}
		/// 		 end()
		///
		/// \attention The return value will always be `end()` if the input value was a null toml::node_view,
		/// 		   because no insertion can take place. This is the only circumstance in which this can occur.
		template <typename ElemType>
		iterator insert(const_iterator pos, ElemType&& val, value_flags flags = preserve_source_value_flags)
		{
			if constexpr (is_node_view<ElemType>)
			{
				if (!val)
					return end();
			}
			return iterator{ insert_at(const_vector_iterator{ pos },
									   impl::make_node(static_cast<ElemType&&>(val), flags)) };
		}

		/// \brief	Repeatedly inserts a new element starting at a specific position in the array.
		///
		/// \detail \cpp
		/// auto arr = toml::array{
		///		"with an evil twinkle in its eye the goose said",
		///		"and immediately we knew peace was never an option."
		///	};
		///	arr.insert(arr.cbegin() + 1, 3, "honk");
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [
		/// 	'with an evil twinkle in its eye the goose said',
		/// 	'honk',
		/// 	'honk',
		/// 	'honk',
		/// 	'and immediately we knew peace was never an option.'
		/// ]
		/// \eout
		///
		/// \tparam ElemType	toml::node, toml::node_view, toml::table, toml::array, or a native TOML value type
		/// 					(or a type promotable to one).
		/// \param 	pos			The insertion position.
		/// \param 	count		The number of times the node or value should be inserted.
		/// \param 	val			The node or value being inserted.
		/// \param	flags		Value flags to apply to new values.
		///
		/// \returns \conditional_return{Valid input}
		/// 		 An iterator to the newly-inserted element.
		/// 		 \conditional_return{count == 0}
		/// 		 A copy of pos
		/// 		 \conditional_return{Input is a null toml::node_view}
		/// 		 end()
		///
		/// \attention The return value will always be `end()` if the input value was a null toml::node_view,
		/// 		   because no insertion can take place. This is the only circumstance in which this can occur.
		template <typename ElemType>
		iterator insert(const_iterator pos,
						size_t count,
						ElemType&& val,
						value_flags flags = preserve_source_value_flags)
		{
			if constexpr (is_node_view<ElemType>)
			{
				if (!val)
					return end();
			}
			switch (count)
			{
				case 0: return iterator{ elems_.begin() + (const_vector_iterator{ pos } - elems_.cbegin()) };
				case 1: return insert(pos, static_cast<ElemType&&>(val), flags);
				default:
				{
					const auto start_idx = static_cast<size_t>(const_vector_iterator{ pos } - elems_.cbegin());
					preinsertion_resize(start_idx, count);
					size_t i = start_idx;
					for (size_t e = start_idx + count - 1u; i < e; i++)
						elems_[i] = impl::make_node(val, flags);

					//# potentially move the initial value into the last element
					elems_[i] = impl::make_node(static_cast<ElemType&&>(val), flags);
					return iterator{ elems_.begin() + static_cast<ptrdiff_t>(start_idx) };
				}
			}
		}

		/// \brief	Inserts a range of elements into the array at a specific position.
		///
		/// \tparam	Iter	An iterator type. Must satisfy ForwardIterator.
		/// \param 	pos		The insertion position.
		/// \param 	first	Iterator to the first node or value being inserted.
		/// \param 	last	Iterator to the one-past-the-last node or value being inserted.
		/// \param	flags		Value flags to apply to new values.
		///
		/// \returns \conditional_return{Valid input}
		/// 		 An iterator to the first newly-inserted element.
		/// 		 \conditional_return{first >= last}
		/// 		 A copy of pos
		/// 		 \conditional_return{All objects in the range were null toml::node_views}
		/// 		 A copy of pos
		template <typename Iter>
		iterator insert(const_iterator pos, Iter first, Iter last, value_flags flags = preserve_source_value_flags)
		{
			const auto distance = std::distance(first, last);
			if (distance <= 0)
				return iterator{ elems_.begin() + (const_vector_iterator{ pos } - elems_.cbegin()) };
			else
			{
				auto count		 = distance;
				using deref_type = decltype(*first);
				if constexpr (is_node_view<deref_type>)
				{
					for (auto it = first; it != last; it++)
						if (!(*it))
							count--;
					if (!count)
						return iterator{ elems_.begin() + (const_vector_iterator{ pos } - elems_.cbegin()) };
				}
				const auto start_idx = static_cast<size_t>(const_vector_iterator{ pos } - elems_.cbegin());
				preinsertion_resize(start_idx, static_cast<size_t>(count));
				size_t i = start_idx;
				for (auto it = first; it != last; it++)
				{
					if constexpr (is_node_view<deref_type>)
					{
						if (!(*it))
							continue;
					}
					if constexpr (std::is_rvalue_reference_v<deref_type>)
						elems_[i++] = impl::make_node(std::move(*it), flags);
					else
						elems_[i++] = impl::make_node(*it, flags);
				}
				return iterator{ elems_.begin() + static_cast<ptrdiff_t>(start_idx) };
			}
		}

		/// \brief	Inserts a range of elements into the array at a specific position.
		///
		/// \tparam ElemType	toml::node_view, toml::table, toml::array, or a native TOML value type
		/// 					(or a type promotable to one).
		/// \param 	pos			The insertion position.
		/// \param 	ilist		An initializer list containing the values to be inserted.
		/// \param	flags		Value flags to apply to new values.
		///
		/// \returns \conditional_return{Valid input}
		///			 An iterator to the first newly-inserted element.
		/// 		 \conditional_return{Input list is empty}
		///			 A copy of pos
		/// 		 \conditional_return{All objects in the list were null toml::node_views}
		///			 A copy of pos
		template <typename ElemType>
		iterator insert(const_iterator pos,
						std::initializer_list<ElemType> ilist,
						value_flags flags = preserve_source_value_flags)
		{
			return insert(pos, ilist.begin(), ilist.end(), flags);
		}

		/// \brief	Emplaces a new element at a specific position in the array.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 1, 2 };
		///
		///	//add a string using std::string's substring constructor
		///	arr.emplace<std::string>(arr.cbegin() + 1, "this is not a drill"sv, 14, 5);
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 'drill', 2 ]
		/// \eout
		///
		/// \tparam ElemType	toml::table, toml::array, or any native TOML value type.
		/// \tparam	Args		Value constructor argument types.
		/// \param 	pos			The insertion position.
		/// \param 	args		Arguments to forward to the value's constructor.
		///
		/// \returns	An iterator to the inserted element.
		///
		/// \remarks There is no difference between insert() and emplace()
		/// 		 for trivial value types (floats, ints, bools).
		template <typename ElemType, typename... Args>
		iterator emplace(const_iterator pos, Args&&... args)
		{
			using type = impl::unwrap_node<ElemType>;
			static_assert((impl::is_native<type> || impl::is_one_of<type, table, array>)&&!impl::is_cvref<type>,
						  "Emplacement type parameter must be one of:" TOML_SA_UNWRAPPED_NODE_TYPE_LIST);

			return iterator{ insert_at(const_vector_iterator{ pos },
									   impl::node_ptr{ new impl::wrap_node<type>{ static_cast<Args&&>(args)... } }) };
		}

		/// \brief	Replaces the element at a specific position in the array with a different value.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 1, 2, 3 };
		/// std::cout << arr << "\n";
		///	arr.replace(arr.cbegin() + 1, "two");
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 2, 3 ]
		/// [ 1, 'two', 3 ]
		/// \eout
		///
		/// \tparam ElemType	toml::node, toml::node_view, toml::table, toml::array, or a native TOML value type
		/// 					(or a type promotable to one).
		/// \param 	pos			The insertion position.
		/// \param 	val			The node or value being inserted.
		/// \param	flags		Value flags to apply to new values.
		///
		/// \returns \conditional_return{Valid input}
		///			 An iterator to the replaced element.
		///			 \conditional_return{Input is a null toml::node_view}
		/// 		 end()
		///
		/// \attention The return value will always be `end()` if the input value was a null toml::node_view,
		/// 		   because no replacement can take place. This is the only circumstance in which this can occur.
		template <typename ElemType>
		iterator replace(const_iterator pos, ElemType&& val, value_flags flags = preserve_source_value_flags)
		{
			TOML_ASSERT(pos >= cbegin() && pos < cend());

			if constexpr (is_node_view<ElemType>)
			{
				if (!val)
					return end();
			}

			const auto it = elems_.begin() + (const_vector_iterator{ pos } - elems_.cbegin());
			*it			  = impl::make_node(static_cast<ElemType&&>(val), flags);
			return iterator{ it };
		}

		/// \brief	Appends a new element to the end of the array.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 1, 2 };
		///	arr.push_back(3);
		///	arr.push_back(4.0);
		///	arr.push_back(toml::array{ 5, "six"sv });
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 2, 3, 4.0, [ 5, 'six' ] ]
		/// \eout
		///
		/// \tparam ElemType	toml::node, toml::node_view, toml::table, toml::array, or a native TOML value type
		/// \param 	val			The node or value being added.
		/// \param	flags		Value flags to apply to new values.
		///
		/// \attention	No insertion takes place if the input value is a null toml::node_view.
		/// 			This is the only circumstance in which this can occur.
		template <typename ElemType>
		void push_back(ElemType&& val, value_flags flags = preserve_source_value_flags)
		{
			emplace_back_if_not_empty_view(static_cast<ElemType&&>(val), flags);
		}

		/// \brief	Emplaces a new element at the end of the array.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 1, 2 };
		///	arr.emplace_back<toml::array>(3, "four"sv);
		/// std::cout << arr << "\n";
		/// \ecpp
		///
		/// \out
		/// [ 1, 2, [ 3, 'four' ] ]
		/// \eout
		///
		/// \tparam ElemType	toml::table, toml::array, or a native TOML value type
		/// \tparam	ElemArgs	Element constructor argument types.
		/// \param 	args		Arguments to forward to the elements's constructor.
		///
		/// \returns A reference to the newly-constructed element.
		///
		/// \remarks There is no difference between push_back() and emplace_back()
		/// 		 For trivial value types (floats, ints, bools).
		template <typename ElemType, typename... ElemArgs>
		decltype(auto) emplace_back(ElemArgs&&... args)
		{
			static_assert(!impl::is_cvref<ElemType>, "ElemType may not be const, volatile, or a reference.");

			static constexpr auto moving_node_ptr = std::is_same_v<ElemType, impl::node_ptr> //
												 && sizeof...(ElemArgs) == 1u				 //
												 && impl::first_is_same<impl::node_ptr&&, ElemArgs&&...>;

			using unwrapped_type = impl::unwrap_node<ElemType>;

			static_assert(
				moving_node_ptr										  //
					|| impl::is_native<unwrapped_type>				  //
					|| impl::is_one_of<unwrapped_type, table, array>, //
				"ElemType argument of array::emplace_back() must be one of:" TOML_SA_UNWRAPPED_NODE_TYPE_LIST);

			if constexpr (moving_node_ptr)
			{
				insert_at_back(static_cast<ElemArgs&&>(args)...);
				return *elems_.back();
			}
			else
			{
				auto ptr = new impl::wrap_node<unwrapped_type>{ static_cast<ElemArgs&&>(args)... };
				insert_at_back(impl::node_ptr{ ptr });
				return *ptr;
			}
		}

		/// @}

	  private:
		/// \cond

		TOML_NODISCARD
		TOML_EXPORTED_STATIC_FUNCTION
		static bool equal(const array&, const array&) noexcept;

		template <typename T>
		TOML_NODISCARD
		static bool equal_to_container(const array& lhs, const T& rhs) noexcept
		{
			using element_type = std::remove_const_t<typename T::value_type>;
			static_assert(impl::is_losslessly_convertible_to_native<element_type>,
						  "Container element type must be losslessly convertible one of the native TOML value types");

			if (lhs.size() != rhs.size())
				return false;
			if (rhs.size() == 0u)
				return true;

			size_t i{};
			for (auto& list_elem : rhs)
			{
				const auto elem = lhs.get_as<impl::native_type_of<element_type>>(i++);
				if (!elem || *elem != list_elem)
					return false;
			}

			return true;
		}

		/// \endcond

	  public:
		/// \name Equality
		/// @{

		/// \brief	Equality operator.
		///
		/// \param 	lhs	The LHS array.
		/// \param 	rhs	The RHS array.
		///
		/// \returns	True if the arrays contained the same elements.
		TOML_NODISCARD
		friend bool operator==(const array& lhs, const array& rhs) noexcept
		{
			return equal(lhs, rhs);
		}

		/// \brief	Inequality operator.
		///
		/// \param 	lhs	The LHS array.
		/// \param 	rhs	The RHS array.
		///
		/// \returns	True if the arrays did not contain the same elements.
		TOML_NODISCARD
		friend bool operator!=(const array& lhs, const array& rhs) noexcept
		{
			return !equal(lhs, rhs);
		}

		/// \brief	Initializer list equality operator.
		template <typename T>
		TOML_NODISCARD
		friend bool operator==(const array& lhs, const std::initializer_list<T>& rhs) noexcept
		{
			return equal_to_container(lhs, rhs);
		}
		TOML_ASYMMETRICAL_EQUALITY_OPS(const array&, const std::initializer_list<T>&, template <typename T>);

		/// \brief	Vector equality operator.
		template <typename T>
		TOML_NODISCARD
		friend bool operator==(const array& lhs, const std::vector<T>& rhs) noexcept
		{
			return equal_to_container(lhs, rhs);
		}
		TOML_ASYMMETRICAL_EQUALITY_OPS(const array&, const std::vector<T>&, template <typename T>);

		/// @}

#if TOML_ENABLE_FORMATTERS

		/// \brief	Prints the array out to a stream as formatted TOML.
		///
		/// \availability This operator is only available when #TOML_ENABLE_FORMATTERS is enabled.
		friend std::ostream& operator<<(std::ostream& lhs, const array& rhs)
		{
			impl::print_to_stream(lhs, rhs);
			return lhs;
		}

#endif
	};
}
TOML_NAMESPACE_END;

#include "header_end.h"
