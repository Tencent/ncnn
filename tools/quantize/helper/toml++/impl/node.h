//# This file is a part of toml++ and is subject to the the terms of the MIT license.
//# Copyright (c) Mark Gillard <mark.gillard@outlook.com.au>
//# See https://github.com/marzer/tomlplusplus/blob/master/LICENSE for the full license text.
// SPDX-License-Identifier: MIT
#pragma once

#include "std_utility.h"
#include "forward_declarations.h"
#include "source_region.h"
#include "header_start.h"

TOML_NAMESPACE_START
{
	/// \brief	A TOML node.
	///
	/// \detail A parsed TOML document forms a tree made up of tables, arrays and values.
	/// 		This type is the base of each of those, providing a lot of the polymorphic plumbing.
	class TOML_ABSTRACT_BASE TOML_EXPORTED_CLASS node
	{
	  private:
		/// \cond

		friend class TOML_PARSER_TYPENAME;
		source_region source_{};

		template <typename T>
		TOML_NODISCARD
		decltype(auto) get_value_exact() const noexcept(impl::value_retrieval_is_nothrow<T>);

		template <typename T, typename N>
		using ref_type_ = std::conditional_t<													//
			std::is_reference_v<T>,																//
			impl::copy_ref<impl::copy_cv<impl::unwrap_node<T>, std::remove_reference_t<N>>, T>, //
			impl::copy_cvref<impl::unwrap_node<T>, N>											//
			>;

		template <typename T, typename N>
		using ref_type = std::conditional_t<			 //
			std::is_reference_v<N>,						 //
			ref_type_<T, N>,							 //
			ref_type_<T, std::add_lvalue_reference_t<N>> //
			>;

		template <typename T, typename N>
		TOML_PURE_GETTER
		static ref_type<T, N&&> do_ref(N&& n) noexcept
		{
			using unwrapped_type = impl::unwrap_node<T>;
			static_assert(toml::is_value<unwrapped_type> || toml::is_container<unwrapped_type>,
						  "The template type argument of node::ref() must be one of:" TOML_SA_UNWRAPPED_NODE_TYPE_LIST);

			TOML_ASSERT_ASSUME(
				n.template is<unwrapped_type>()
				&& "template type argument provided to toml::node::ref() didn't match the node's actual type");

			using node_ref = std::remove_volatile_t<std::remove_reference_t<N>>&;
			using val_type = std::remove_volatile_t<unwrapped_type>;
			using out_ref  = ref_type<T, N&&>;
			static_assert(std::is_reference_v<out_ref>);

			if constexpr (toml::is_value<unwrapped_type>)
				return static_cast<out_ref>(const_cast<node_ref>(n).template ref_cast<val_type>().get());
			else
				return static_cast<out_ref>(const_cast<node_ref>(n).template ref_cast<val_type>());
		}

	  protected:
		TOML_EXPORTED_MEMBER_FUNCTION
		node() noexcept;

		TOML_EXPORTED_MEMBER_FUNCTION
		node(const node&) noexcept;

		TOML_EXPORTED_MEMBER_FUNCTION
		node(node&&) noexcept;

		TOML_EXPORTED_MEMBER_FUNCTION
		node& operator=(const node&) noexcept;

		TOML_EXPORTED_MEMBER_FUNCTION
		node& operator=(node&&) noexcept;

		template <typename T, typename N>
		using ref_cast_type_ = std::conditional_t<											  //
			std::is_reference_v<T>,															  //
			impl::copy_ref<impl::copy_cv<impl::wrap_node<T>, std::remove_reference_t<N>>, T>, //
			impl::copy_cvref<impl::wrap_node<T>, N>											  //
			>;

		template <typename T, typename N>
		using ref_cast_type = std::conditional_t<			  //
			std::is_reference_v<N>,							  //
			ref_cast_type_<T, N>,							  //
			ref_cast_type_<T, std::add_lvalue_reference_t<N>> //
			>;

		template <typename T>
		TOML_PURE_INLINE_GETTER
		ref_cast_type<T, node&> ref_cast() & noexcept
		{
			using out_ref  = ref_cast_type<T, node&>;
			using out_type = std::remove_reference_t<out_ref>;
			return static_cast<out_ref>(*reinterpret_cast<out_type*>(this));
		}

		template <typename T>
		TOML_PURE_INLINE_GETTER
		ref_cast_type<T, node&&> ref_cast() && noexcept
		{
			using out_ref  = ref_cast_type<T, node&&>;
			using out_type = std::remove_reference_t<out_ref>;
			return static_cast<out_ref>(*reinterpret_cast<out_type*>(this));
		}

		template <typename T>
		TOML_PURE_INLINE_GETTER
		ref_cast_type<T, const node&> ref_cast() const& noexcept
		{
			using out_ref  = ref_cast_type<T, const node&>;
			using out_type = std::remove_reference_t<out_ref>;
			return static_cast<out_ref>(*reinterpret_cast<out_type*>(this));
		}

		template <typename T>
		TOML_PURE_INLINE_GETTER
		ref_cast_type<T, const node&&> ref_cast() const&& noexcept
		{
			using out_ref  = ref_cast_type<T, const node&&>;
			using out_type = std::remove_reference_t<out_ref>;
			return static_cast<out_ref>(*reinterpret_cast<out_type*>(this));
		}

		/// \endcond

	  public:
		TOML_EXPORTED_MEMBER_FUNCTION
		virtual ~node() noexcept;

		/// \name Type checks
		/// @{

		/// \brief	Checks if a node contains values/elements of only one type.
		///
		/// \detail \cpp
		/// auto cfg = toml::parse("arr = [ 1, 2, 3, 4.0 ]");
		/// toml::array& arr = *cfg["arr"].as_array();
		///
		/// toml::node* nonmatch{};
		/// if (arr.is_homogeneous(toml::node_type::integer, nonmatch))
		/// 	std::cout << "array was homogeneous"sv << "\n";
		/// else
		/// 	std::cout << "array was not homogeneous!\n"
		/// 	<< "first non-match was a "sv << nonmatch->type() << " at " << nonmatch->source() << "\n";
		/// \ecpp
		///
		/// \out
		/// array was not homogeneous!
		///	first non-match was a floating-point at line 1, column 18
		/// \eout
		///
		/// \param	ntype	A TOML node type. <br>
		/// 				\conditional_return{toml::node_type::none}
		///					"is every element the same type?"
		/// 				\conditional_return{Anything else}
		///					"is every element one of these?"
		///
		/// \param first_nonmatch	Reference to a pointer in which the address of the first non-matching element
		/// 						will be stored if the return value is false.
		///
		/// \returns	True if the node was homogeneous.
		///
		/// \remarks	Always returns `false` for empty tables and arrays.
		TOML_PURE_GETTER
		virtual bool is_homogeneous(node_type ntype, node*& first_nonmatch) noexcept = 0;

		/// \brief	Checks if a node contains values/elements of only one type (const overload).
		TOML_PURE_GETTER
		virtual bool is_homogeneous(node_type ntype, const node*& first_nonmatch) const noexcept = 0;

		/// \brief	Checks if the node contains values/elements of only one type.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 1, 2, 3 };
		/// std::cout << "homogenous: "sv << arr.is_homogeneous(toml::node_type::none) << "\n";
		/// std::cout << "all floats: "sv << arr.is_homogeneous(toml::node_type::floating_point) << "\n";
		/// std::cout << "all arrays: "sv << arr.is_homogeneous(toml::node_type::array) << "\n";
		/// std::cout << "all ints:   "sv << arr.is_homogeneous(toml::node_type::integer) << "\n";
		/// \ecpp
		///
		/// \out
		/// homogeneous: true
		/// all floats:  false
		/// all arrays:  false
		/// all ints:    true
		/// \eout
		///
		/// \param	ntype	A TOML node type. <br>
		/// 				\conditional_return{toml::node_type::none}
		///					"is every element the same type?"
		/// 				\conditional_return{Anything else}
		///					"is every element one of these?"
		///
		/// \returns	True if the node was homogeneous.
		///
		/// \remarks	Always returns `false` for empty tables and arrays.
		TOML_PURE_GETTER
		virtual bool is_homogeneous(node_type ntype) const noexcept = 0;

		/// \brief	Checks if the node contains values/elements of only one type.
		///
		/// \detail \cpp
		/// auto arr = toml::array{ 1, 2, 3 };
		/// std::cout << "homogenous:   "sv << arr.is_homogeneous() << "\n";
		/// std::cout << "all doubles:  "sv << arr.is_homogeneous<double>() << "\n";
		/// std::cout << "all arrays:   "sv << arr.is_homogeneous<toml::array>() << "\n";
		/// std::cout << "all integers: "sv << arr.is_homogeneous<int64_t>() << "\n";
		/// \ecpp
		///
		/// \out
		/// homogeneous: true
		/// all floats:  false
		/// all arrays:  false
		/// all ints:    true
		/// \eout
		///
		/// \tparam	ElemType	A TOML node or value type. <br>
		/// 					\conditional_return{Left as `void`}
		///						"is every element the same type?" <br>
		/// 					\conditional_return{Explicitly specified}
		///						"is every element a T?"
		///
		/// \returns	True if the node was homogeneous.
		///
		/// \remarks	Always returns `false` for empty tables and arrays.
		template <typename ElemType = void>
		TOML_PURE_GETTER
		bool is_homogeneous() const noexcept
		{
			using unwrapped_type = impl::unwrap_node<impl::remove_cvref<ElemType>>;
			static_assert(std::is_void_v<unwrapped_type> //
							  || (toml::is_value<unwrapped_type> || toml::is_container<unwrapped_type>),
						  "The template type argument of node::is_homogeneous() must be void or one "
						  "of:" TOML_SA_UNWRAPPED_NODE_TYPE_LIST);

			return is_homogeneous(impl::node_type_of<unwrapped_type>);
		}

		/// \brief	Returns the node's type identifier.
		TOML_PURE_GETTER
		virtual node_type type() const noexcept = 0;

		/// \brief	Returns true if this node is a table.
		TOML_PURE_GETTER
		virtual bool is_table() const noexcept = 0;

		/// \brief	Returns true if this node is an array.
		TOML_PURE_GETTER
		virtual bool is_array() const noexcept = 0;

		/// \brief	Returns true if this node is an array containing only tables.
		TOML_PURE_GETTER
		virtual bool is_array_of_tables() const noexcept = 0;

		/// \brief	Returns true if this node is a value.
		TOML_PURE_GETTER
		virtual bool is_value() const noexcept = 0;

		/// \brief	Returns true if this node is a string value.
		TOML_PURE_GETTER
		virtual bool is_string() const noexcept = 0;

		/// \brief	Returns true if this node is an integer value.
		TOML_PURE_GETTER
		virtual bool is_integer() const noexcept = 0;

		/// \brief	Returns true if this node is an floating-point value.
		TOML_PURE_GETTER
		virtual bool is_floating_point() const noexcept = 0;

		/// \brief	Returns true if this node is an integer or floating-point value.
		TOML_PURE_GETTER
		virtual bool is_number() const noexcept = 0;

		/// \brief	Returns true if this node is a boolean value.
		TOML_PURE_GETTER
		virtual bool is_boolean() const noexcept = 0;

		/// \brief	Returns true if this node is a local date value.
		TOML_PURE_GETTER
		virtual bool is_date() const noexcept = 0;

		/// \brief	Returns true if this node is a local time value.
		TOML_PURE_GETTER
		virtual bool is_time() const noexcept = 0;

		/// \brief	Returns true if this node is a date-time value.
		TOML_PURE_GETTER
		virtual bool is_date_time() const noexcept = 0;

		/// \brief	Checks if a node is a specific type.
		///
		/// \tparam	T	A TOML node or value type.
		///
		/// \returns	Returns true if this node is an instance of the specified type.
		template <typename T>
		TOML_PURE_INLINE_GETTER
		bool is() const noexcept
		{
			using unwrapped_type = impl::unwrap_node<impl::remove_cvref<T>>;
			static_assert(toml::is_value<unwrapped_type> || toml::is_container<unwrapped_type>,
						  "The template type argument of node::is() must be one of:" TOML_SA_UNWRAPPED_NODE_TYPE_LIST);

			if constexpr (std::is_same_v<unwrapped_type, table>)
				return is_table();
			else if constexpr (std::is_same_v<unwrapped_type, array>)
				return is_array();
			else if constexpr (std::is_same_v<unwrapped_type, std::string>)
				return is_string();
			else if constexpr (std::is_same_v<unwrapped_type, int64_t>)
				return is_integer();
			else if constexpr (std::is_same_v<unwrapped_type, double>)
				return is_floating_point();
			else if constexpr (std::is_same_v<unwrapped_type, bool>)
				return is_boolean();
			else if constexpr (std::is_same_v<unwrapped_type, date>)
				return is_date();
			else if constexpr (std::is_same_v<unwrapped_type, time>)
				return is_time();
			else if constexpr (std::is_same_v<unwrapped_type, date_time>)
				return is_date_time();
		}

		/// @}

		/// \name Type casts
		/// @{

		/// \brief	Returns a pointer to the node as a toml::table, if it is one.
		TOML_PURE_GETTER
		virtual table* as_table() noexcept = 0;

		/// \brief	Returns a pointer to the node as a toml::array, if it is one.
		TOML_PURE_GETTER
		virtual array* as_array() noexcept = 0;

		/// \brief	Returns a pointer to the node as a toml::value<std::string>, if it is one.
		TOML_PURE_GETTER
		virtual toml::value<std::string>* as_string() noexcept = 0;

		/// \brief	Returns a pointer to the node as a toml::value<int64_t>, if it is one.
		TOML_PURE_GETTER
		virtual toml::value<int64_t>* as_integer() noexcept = 0;

		/// \brief	Returns a pointer to the node as a toml::value<double>, if it is one.
		TOML_PURE_GETTER
		virtual toml::value<double>* as_floating_point() noexcept = 0;

		/// \brief	Returns a pointer to the node as a toml::value<bool>, if it is one.
		TOML_PURE_GETTER
		virtual toml::value<bool>* as_boolean() noexcept = 0;

		/// \brief	Returns a pointer to the node as a toml::value<toml::date>, if it is one.
		TOML_PURE_GETTER
		virtual toml::value<date>* as_date() noexcept = 0;

		/// \brief	Returns a pointer to the node as a toml::value<toml::time>, if it is one.
		TOML_PURE_GETTER
		virtual toml::value<time>* as_time() noexcept = 0;

		/// \brief	Returns a pointer to the node as a toml::value<toml::date_time>, if it is one.
		TOML_PURE_GETTER
		virtual toml::value<date_time>* as_date_time() noexcept = 0;

		/// \brief	Returns a const-qualified pointer to the node as a toml::table, if it is one.
		TOML_PURE_GETTER
		virtual const table* as_table() const noexcept = 0;

		/// \brief	Returns a const-qualified pointer to the node as a toml::array, if it is one.
		TOML_PURE_GETTER
		virtual const array* as_array() const noexcept = 0;

		/// \brief	Returns a const-qualified pointer to the node as a toml::value<std::string>, if it is one.
		TOML_PURE_GETTER
		virtual const toml::value<std::string>* as_string() const noexcept = 0;

		/// \brief	Returns a const-qualified pointer to the node as a toml::value<int64_t>, if it is one.
		TOML_PURE_GETTER
		virtual const toml::value<int64_t>* as_integer() const noexcept = 0;

		/// \brief	Returns a const-qualified pointer to the node as a toml::value<double>, if it is one.
		TOML_PURE_GETTER
		virtual const toml::value<double>* as_floating_point() const noexcept = 0;

		/// \brief	Returns a const-qualified pointer to the node as a toml::value<bool>, if it is one.
		TOML_PURE_GETTER
		virtual const toml::value<bool>* as_boolean() const noexcept = 0;

		/// \brief	Returns a const-qualified pointer to the node as a toml::value<toml::date>, if it is one.
		TOML_PURE_GETTER
		virtual const toml::value<date>* as_date() const noexcept = 0;

		/// \brief	Returns a const-qualified pointer to the node as a toml::value<toml::time>, if it is one.
		TOML_PURE_GETTER
		virtual const toml::value<time>* as_time() const noexcept = 0;

		/// \brief	Returns a const-qualified pointer to the node as a toml::value<toml::date_time>, if it is one.
		TOML_PURE_GETTER
		virtual const toml::value<date_time>* as_date_time() const noexcept = 0;

		/// \brief	Gets a pointer to the node as a more specific node type.
		///
		/// \details \cpp
		///
		/// toml::value<int64_t>* int_value = node->as<int64_t>();
		/// toml::table* tbl = node->as<toml::table>();
		/// if (int_value)
		///		std::cout << "Node is a value<int64_t>\n";
		/// else if (tbl)
		///		std::cout << "Node is a table\n";
		///
		///	// fully-qualified value node types also work (useful for template code):
		///	toml::value<int64_t>* int_value2 = node->as<toml::value<int64_t>>();
		/// if (int_value2)
		///		std::cout << "Node is a value<int64_t>\n";
		/// \ecpp
		///
		/// \tparam	T	The node type or TOML value type to cast to.
		///
		/// \returns	A pointer to the node as the given type, or nullptr if it was a different type.
		template <typename T>
		TOML_PURE_INLINE_GETTER
		impl::wrap_node<T>* as() noexcept
		{
			using unwrapped_type = impl::unwrap_node<impl::remove_cvref<T>>;
			static_assert(toml::is_value<unwrapped_type> || toml::is_container<unwrapped_type>,
						  "The template type argument of node::as() must be one of:" TOML_SA_UNWRAPPED_NODE_TYPE_LIST);

			if constexpr (std::is_same_v<unwrapped_type, table>)
				return as_table();
			else if constexpr (std::is_same_v<unwrapped_type, array>)
				return as_array();
			else if constexpr (std::is_same_v<unwrapped_type, std::string>)
				return as_string();
			else if constexpr (std::is_same_v<unwrapped_type, int64_t>)
				return as_integer();
			else if constexpr (std::is_same_v<unwrapped_type, double>)
				return as_floating_point();
			else if constexpr (std::is_same_v<unwrapped_type, bool>)
				return as_boolean();
			else if constexpr (std::is_same_v<unwrapped_type, date>)
				return as_date();
			else if constexpr (std::is_same_v<unwrapped_type, time>)
				return as_time();
			else if constexpr (std::is_same_v<unwrapped_type, date_time>)
				return as_date_time();
		}

		/// \brief	Gets a pointer to the node as a more specific node type (const overload).
		template <typename T>
		TOML_PURE_INLINE_GETTER
		const impl::wrap_node<T>* as() const noexcept
		{
			using unwrapped_type = impl::unwrap_node<impl::remove_cvref<T>>;
			static_assert(toml::is_value<unwrapped_type> || toml::is_container<unwrapped_type>,
						  "The template type argument of node::as() must be one of:" TOML_SA_UNWRAPPED_NODE_TYPE_LIST);

			if constexpr (std::is_same_v<unwrapped_type, table>)
				return as_table();
			else if constexpr (std::is_same_v<unwrapped_type, array>)
				return as_array();
			else if constexpr (std::is_same_v<unwrapped_type, std::string>)
				return as_string();
			else if constexpr (std::is_same_v<unwrapped_type, int64_t>)
				return as_integer();
			else if constexpr (std::is_same_v<unwrapped_type, double>)
				return as_floating_point();
			else if constexpr (std::is_same_v<unwrapped_type, bool>)
				return as_boolean();
			else if constexpr (std::is_same_v<unwrapped_type, date>)
				return as_date();
			else if constexpr (std::is_same_v<unwrapped_type, time>)
				return as_time();
			else if constexpr (std::is_same_v<unwrapped_type, date_time>)
				return as_date_time();
		}

		/// @}

		/// \name Value retrieval
		/// @{

		/// \brief	Gets the value contained by this node.
		///
		/// \detail This function has 'exact' retrieval semantics; the only return value types allowed are the
		/// 		TOML native value types, or types that can losslessly represent a native value type (e.g.
		/// 		std::wstring on Windows).
		///
		/// \tparam	T	One of the native TOML value types, or a type capable of losslessly representing one.
		///
		/// \returns	The underlying value if the node was a value of the
		/// 			matching type (or losslessly convertible to it), or an empty optional.
		///
		/// \see node::value()
		template <typename T>
		TOML_NODISCARD
		optional<T> value_exact() const noexcept(impl::value_retrieval_is_nothrow<T>);

		/// \brief	Gets the value contained by this node.
		///
		/// \detail This function has 'permissive' retrieval semantics; some value types are allowed
		/// 		to convert to others (e.g. retrieving a boolean as an integer), and the specified return value
		/// 		type can be any type where a reasonable conversion from a native TOML value exists
		/// 		(e.g. std::wstring on Windows). If the source value cannot be represented by
		/// 		the destination type, an empty optional is returned.
		///
		/// \godbolt{zzG81K}
		///
		/// \cpp
		/// auto tbl = toml::parse(R"(
		/// 	int	= -10
		/// 	flt	= 25.0
		/// 	pi	= 3.14159
		/// 	bool = false
		/// 	huge = 9223372036854775807
		/// 	str	= "foo"
		/// )"sv);
		///
		/// const auto print_value_with_typename =
		/// 	[&](std::string_view key, std::string_view type_name, auto* dummy)
		/// 	{
		/// 		std::cout << "- " << std::setw(18) << std::left << type_name;
		/// 		using type = std::remove_pointer_t<decltype(dummy)>;
		/// 		if (auto val = tbl.get(key)->value<type>(); val)
		/// 			std::cout << *val << "\n";
		/// 		else
		/// 			std::cout << "n/a\n";
		/// 	};
		///
		/// #define print_value(key, T) print_value_with_typename(key, #T, (T*)nullptr)
		///
		/// for (auto key : { "int", "flt", "pi", "bool", "huge", "str" })
		/// {
		/// 	std::cout << tbl[key].type() << " value '" << key << "' as:\n";
		/// 	print_value(key, bool);
		/// 	print_value(key, int);
		/// 	print_value(key, unsigned int);
		/// 	print_value(key, long long);
		/// 	print_value(key, float);
		/// 	print_value(key, double);
		/// 	print_value(key, std::string);
		/// 	print_value(key, std::string_view);
		/// 	print_value(key, const char*);
		/// 	std::cout << "\n";
		/// }
		/// \ecpp
		///
		/// \out
		/// integer value 'int' as:
		/// - bool              true
		/// - int               -10
		/// - unsigned int      n/a
		/// - long long         -10
		/// - float             -10
		/// - double            -10
		/// - std::string       n/a
		/// - std::string_view  n/a
		/// - const char*       n/a
		///
		/// floating-point value 'flt' as:
		/// - bool              n/a
		/// - int               25
		/// - unsigned int      25
		/// - long long         25
		/// - float             25
		/// - double            25
		/// - std::string       n/a
		/// - std::string_view  n/a
		/// - const char*       n/a
		///
		/// floating-point value 'pi' as:
		/// - bool              n/a
		/// - int               n/a
		/// - unsigned int      n/a
		/// - long long         n/a
		/// - float             3.14159
		/// - double            3.14159
		/// - std::string       n/a
		/// - std::string_view  n/a
		/// - const char*       n/a
		///
		/// boolean value 'bool' as:
		/// - bool              false
		/// - int               0
		/// - unsigned int      0
		/// - long long         0
		/// - float             n/a
		/// - double            n/a
		/// - std::string       n/a
		/// - std::string_view  n/a
		/// - const char*       n/a
		///
		/// integer value 'huge' as:
		/// - bool              true
		/// - int               n/a
		/// - unsigned int      n/a
		/// - long long         9223372036854775807
		/// - float             n/a
		/// - double            n/a
		/// - std::string       n/a
		/// - std::string_view  n/a
		/// - const char*       n/a
		///
		/// string value 'str' as:
		/// - bool              n/a
		/// - int               n/a
		/// - unsigned int      n/a
		/// - long long         n/a
		/// - float             n/a
		/// - double            n/a
		/// - std::string       foo
		/// - std::string_view  foo
		/// - const char*       foo
		/// \eout
		///
		/// \tparam	T	One of the native TOML value types, or a type capable of converting to one.
		///
		/// \returns	The underlying value if the node was a value of the matching type (or convertible to it)
		/// 			and within the range of the output type, or an empty optional.
		///
		/// \note		If you want strict value retrieval semantics that do not allow for any type conversions,
		/// 			use node::value_exact() instead.
		///
		/// \see node::value_exact()
		template <typename T>
		TOML_NODISCARD
		optional<T> value() const noexcept(impl::value_retrieval_is_nothrow<T>);

		/// \brief	Gets the raw value contained by this node, or a default.
		///
		/// \tparam	T				Default value type. Must be one of the native TOML value types,
		/// 						or convertible to it.
		/// \param 	default_value	The default value to return if the node wasn't a value, wasn't the
		/// 						correct type, or no conversion was possible.
		///
		/// \returns	The underlying value if the node was a value of the matching type (or convertible to it)
		/// 			and within the range of the output type, or the provided default.
		///
		/// \note	This function has the same permissive retrieval semantics as node::value(). If you want strict
		/// 		value retrieval semantics that do not allow for any type conversions, use node::value_exact()
		/// 		instead.
		///
		/// \see
		/// 	- node::value()
		///		- node::value_exact()
		template <typename T>
		TOML_NODISCARD
		auto value_or(T&& default_value) const noexcept(impl::value_retrieval_is_nothrow<T>);

		/// \brief	Gets a raw reference to a node's underlying data.
		///
		/// \warning This function is dangerous if used carelessly and **WILL** break your code if the
		///			 chosen value type doesn't match the node's actual type. In debug builds an assertion
		///			 will fire when invalid accesses are attempted: \cpp
		///
		/// auto tbl = toml::parse(R"(
		///		min = 32
		///		max = 45
		/// )"sv);
		///
		/// int64_t& min_ref = tbl.at("min").ref<int64_t>(); // matching type
		/// double& max_ref = tbl.at("max").ref<double>();  // mismatched type, hits assert()
		/// \ecpp
		///
		/// \note	Specifying explicit ref qualifiers acts as an explicit ref-category cast,
		///			whereas specifying explicit cv-ref qualifiers merges them with whatever
		///			the cv qualification of the node is (to ensure cv-correctness is propagated), e.g.:
		///			| node        | T                      | return type                  |
		///			|-------------|------------------------|------------------------------|
		///			| node&       | std::string            | std::string&                 |
		///			| node&       | std::string&&          | std::string&&                |
		///			| const node& | volatile std::string   | const volatile std::string&  |
		///			| const node& | volatile std::string&& | const volatile std::string&& |
		///
		/// \tparam	T	toml::table, toml::array, or one of the TOML value types.
		///
		/// \returns	A reference to the underlying data.
		template <typename T>
		TOML_PURE_GETTER
		decltype(auto) ref() & noexcept
		{
			return do_ref<T>(*this);
		}

		/// \brief	Gets a raw reference to a node's underlying data (rvalue overload).
		template <typename T>
		TOML_PURE_GETTER
		decltype(auto) ref() && noexcept
		{
			return do_ref<T>(std::move(*this));
		}

		/// \brief	Gets a raw reference to a node's underlying data (const lvalue overload).
		template <typename T>
		TOML_PURE_GETTER
		decltype(auto) ref() const& noexcept
		{
			return do_ref<T>(*this);
		}

		/// \brief	Gets a raw reference to a node's underlying data (const rvalue overload).
		template <typename T>
		TOML_PURE_GETTER
		decltype(auto) ref() const&& noexcept
		{
			return do_ref<T>(std::move(*this));
		}

		/// @}

		/// \name Metadata
		/// @{

		/// \brief	Returns the source region responsible for generating this node during parsing.
		TOML_PURE_INLINE_GETTER
		const source_region& source() const noexcept
		{
			return source_;
		}

		/// @}

	  private:
		/// \cond

		template <typename Func, typename Node, typename T>
		static constexpr bool can_visit = std::is_invocable_v<Func, ref_cast_type<T, Node>>;

		template <typename Func, typename Node, typename T>
		static constexpr bool can_visit_nothrow = std::is_nothrow_invocable_v<Func, ref_cast_type<T, Node>>;

		template <typename Func, typename Node>
		static constexpr bool can_visit_any = can_visit<Func, Node, table>		 //
										   || can_visit<Func, Node, array>		 //
										   || can_visit<Func, Node, std::string> //
										   || can_visit<Func, Node, int64_t>	 //
										   || can_visit<Func, Node, double>		 //
										   || can_visit<Func, Node, bool>		 //
										   || can_visit<Func, Node, date>		 //
										   || can_visit<Func, Node, time>		 //
										   || can_visit<Func, Node, date_time>;

		// clang-format off

		template <typename Func, typename Node>
		static constexpr bool can_visit_all = can_visit<Func, Node, table>		 //
										   && can_visit<Func, Node, array>		 //
										   && can_visit<Func, Node, std::string> //
										   && can_visit<Func, Node, int64_t>	 //
										   && can_visit<Func, Node, double>		 //
										   && can_visit<Func, Node, bool>		 //
										   && can_visit<Func, Node, date>		 //
										   && can_visit<Func, Node, time>		 //
										   && can_visit<Func, Node, date_time>;

		template <typename Func, typename Node, typename T>
		static constexpr bool visit_is_nothrow_one = !can_visit<Func, Node, T> || can_visit_nothrow<Func, Node, T>;

		template <typename Func, typename Node>
		static constexpr bool visit_is_nothrow = visit_is_nothrow_one<Func, Node, table>	   //
											  && visit_is_nothrow_one<Func, Node, array>	   //
											  && visit_is_nothrow_one<Func, Node, std::string> //
											  && visit_is_nothrow_one<Func, Node, int64_t>	   //
											  && visit_is_nothrow_one<Func, Node, double>	   //
											  && visit_is_nothrow_one<Func, Node, bool>		   //
											  && visit_is_nothrow_one<Func, Node, date>		   //
											  && visit_is_nothrow_one<Func, Node, time>		   //
											  && visit_is_nothrow_one<Func, Node, date_time>;

		// clang-format on

		template <typename Func, typename Node, typename T, bool = can_visit<Func, Node, T>>
		struct visit_return_type_
		{
			using type = decltype(std::declval<Func>()(std::declval<ref_cast_type<T, Node>>()));
		};
		template <typename Func, typename Node, typename T>
		struct visit_return_type_<Func, Node, T, false>
		{
			using type = void;
		};

		template <typename Func, typename Node, typename T>
		using visit_return_type = typename visit_return_type_<Func, Node, T>::type;

		template <typename A, typename B>
		using nonvoid = std::conditional_t<std::is_void_v<A>, B, A>;

		template <typename Func, typename Node>
		static decltype(auto) do_visit(Func&& visitor, Node&& n) noexcept(visit_is_nothrow<Func&&, Node&&>)
		{
			static_assert(can_visit_any<Func&&, Node&&>,
						  "TOML node visitors must be invocable for at least one of the toml::node "
						  "specializations:" TOML_SA_NODE_TYPE_LIST);

			switch (n.type())
			{
				case node_type::table:
					if constexpr (can_visit<Func&&, Node&&, table>)
						return static_cast<Func&&>(visitor)(static_cast<Node&&>(n).template ref_cast<table>());
					break;

				case node_type::array:
					if constexpr (can_visit<Func&&, Node&&, array>)
						return static_cast<Func&&>(visitor)(static_cast<Node&&>(n).template ref_cast<array>());
					break;

				case node_type::string:
					if constexpr (can_visit<Func&&, Node&&, std::string>)
						return static_cast<Func&&>(visitor)(static_cast<Node&&>(n).template ref_cast<std::string>());
					break;

				case node_type::integer:
					if constexpr (can_visit<Func&&, Node&&, int64_t>)
						return static_cast<Func&&>(visitor)(static_cast<Node&&>(n).template ref_cast<int64_t>());
					break;

				case node_type::floating_point:
					if constexpr (can_visit<Func&&, Node&&, double>)
						return static_cast<Func&&>(visitor)(static_cast<Node&&>(n).template ref_cast<double>());
					break;

				case node_type::boolean:
					if constexpr (can_visit<Func&&, Node&&, bool>)
						return static_cast<Func&&>(visitor)(static_cast<Node&&>(n).template ref_cast<bool>());
					break;

				case node_type::date:
					if constexpr (can_visit<Func&&, Node&&, date>)
						return static_cast<Func&&>(visitor)(static_cast<Node&&>(n).template ref_cast<date>());
					break;

				case node_type::time:
					if constexpr (can_visit<Func&&, Node&&, time>)
						return static_cast<Func&&>(visitor)(static_cast<Node&&>(n).template ref_cast<time>());
					break;

				case node_type::date_time:
					if constexpr (can_visit<Func&&, Node&&, date_time>)
						return static_cast<Func&&>(visitor)(static_cast<Node&&>(n).template ref_cast<date_time>());
					break;

				case node_type::none: TOML_UNREACHABLE;
				default: TOML_UNREACHABLE;
			}

			if constexpr (!can_visit_all<Func&&, Node&&>)
			{
				// clang-format off

				using return_type =
					nonvoid<visit_return_type<Func&&, Node&&, table>,
					nonvoid<visit_return_type<Func&&, Node&&, array>,
					nonvoid<visit_return_type<Func&&, Node&&, std::string>,
					nonvoid<visit_return_type<Func&&, Node&&, int64_t>,
					nonvoid<visit_return_type<Func&&, Node&&, double>,
					nonvoid<visit_return_type<Func&&, Node&&, bool>,
					nonvoid<visit_return_type<Func&&, Node&&, date>,
					nonvoid<visit_return_type<Func&&, Node&&, time>,
							visit_return_type<Func&&, Node&&, date_time>
				>>>>>>>>;

				// clang-format on

				if constexpr (!std::is_void_v<return_type>)
				{
					static_assert(std::is_default_constructible_v<return_type>,
								  "Non-exhaustive visitors must return a default-constructible type, or void");
					return return_type{};
				}
			}
		}

		/// \endcond

	  public:
		/// \name Visitation
		/// @{

		/// \brief	Invokes a visitor on the node based on the node's concrete type.
		///
		/// \details Visitation is useful when you expect
		/// 		 a node to be one of a set number of types and need
		/// 		 to handle these types differently. Using `visit()` allows
		/// 		 you to eliminate some of the casting/conversion boilerplate: \cpp
		///
		/// node.visit([](auto&& n)
		/// {
		///		if constexpr (toml::is_string<decltype(n)>)
		///			do_something_with_a_string(*n)); //n is a toml::value<std::string>
		///		else if constexpr (toml::is_integer<decltype(n)>)
		///			do_something_with_an_int(*n); //n is a toml::value<int64_t>
		/// });
		/// \ecpp
		///
		/// Visitor functions need not be generic; specifying a concrete node type as the input argument type
		/// effectively acts a 'filter', only invoking the visitor if the concrete type is compatible.
		/// Thus the example above can be re-written as: \cpp
		/// node.visit([](toml::value<std::string>& s) { do_something_with_a_string(*s)); });
		/// node.visit([](toml::value<int64_t>& i)     { do_something_with_an_int(*i)); });
		/// \ecpp
		///
		/// \tparam	Func	A callable type invocable with one or more of the toml++ node types.
		///
		/// \param 	visitor	The visitor object.
		///
		/// \returns The return value of the visitor.
		/// 		 Can be void. Non-exhaustive visitors must return a default-constructible type.
		///
		/// \see https://en.wikipedia.org/wiki/Visitor_pattern
		template <typename Func>
		decltype(auto) visit(Func&& visitor) & noexcept(visit_is_nothrow<Func&&, node&>)
		{
			return do_visit(static_cast<Func&&>(visitor), *this);
		}

		/// \brief	Invokes a visitor on the node based on the node's concrete type (rvalue overload).
		template <typename Func>
		decltype(auto) visit(Func&& visitor) && noexcept(visit_is_nothrow<Func&&, node&&>)
		{
			return do_visit(static_cast<Func&&>(visitor), static_cast<node&&>(*this));
		}

		/// \brief	Invokes a visitor on the node based on the node's concrete type (const lvalue overload).
		template <typename Func>
		decltype(auto) visit(Func&& visitor) const& noexcept(visit_is_nothrow<Func&&, const node&>)
		{
			return do_visit(static_cast<Func&&>(visitor), *this);
		}

		/// \brief	Invokes a visitor on the node based on the node's concrete type (const rvalue overload).
		template <typename Func>
		decltype(auto) visit(Func&& visitor) const&& noexcept(visit_is_nothrow<Func&&, const node&&>)
		{
			return do_visit(static_cast<Func&&>(visitor), static_cast<const node&&>(*this));
		}

		/// @}

		/// \name Node views
		/// @{

		/// \brief	Creates a node_view pointing to this node.
		TOML_NODISCARD
		explicit operator node_view<node>() noexcept;

		/// \brief	Creates a node_view pointing to this node (const overload).
		TOML_NODISCARD
		explicit operator node_view<const node>() const noexcept;

		/// \brief Returns a view of the subnode matching a fully-qualified "TOML path".
		///
		/// \detail \cpp
		/// auto config = toml::parse(R"(
		///
		/// [foo]
		/// bar = [ 0, 1, 2, [ 3 ], { kek = 4 } ]
		///
		/// )"sv);
		///
		/// std::cout << config.at_path("foo.bar[2]") << "\n";
		/// std::cout << config.at_path("foo.bar[3][0]") << "\n";
		/// std::cout << config.at_path("foo.bar[4].kek") << "\n";
		/// \ecpp
		///
		/// \out
		/// 2
		/// 3
		/// 4
		/// \eout
		///
		///
		/// \note Keys in paths are interpreted literally, so whitespace (or lack thereof) matters:
		/// \cpp
		/// config.at_path( "foo.bar")  // same as node_view{ config }["foo"]["bar"]
		/// config.at_path( "foo. bar") // same as node_view{ config }["foo"][" bar"]
		/// config.at_path( "foo..bar") // same as node_view{ config }["foo"][""]["bar"]
		/// config.at_path( ".foo.bar") // same as node_view{ config }[""]["foo"]["bar"]
		/// \ecpp
		/// <br>
		/// Additionally, TOML allows '.' (period) characters to appear in keys if they are quoted strings.
		/// This function makes no allowance for this, instead treating all period characters as sub-table delimiters.
		/// If you have periods in your table keys, first consider:
		/// 1. Not doing that
		/// 2. Using node_view::operator[] instead.
		///
		/// \param path		The "TOML path" to traverse.
		TOML_NODISCARD
		TOML_EXPORTED_MEMBER_FUNCTION
		node_view<node> at_path(std::string_view path) noexcept;

		/// \brief Returns a const view of the subnode matching a fully-qualified "TOML path".
		///
		/// \see #at_path(std::string_view)
		TOML_NODISCARD
		TOML_EXPORTED_MEMBER_FUNCTION
		node_view<const node> at_path(std::string_view path) const noexcept;

#if TOML_ENABLE_WINDOWS_COMPAT

		/// \brief Returns a view of the subnode matching a fully-qualified "TOML path".
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \see #at_path(std::string_view)
		TOML_NODISCARD
		TOML_EXPORTED_MEMBER_FUNCTION
		node_view<node> at_path(std::wstring_view path);

		/// \brief Returns a const view of the subnode matching a fully-qualified "TOML path".
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \see #at_path(std::string_view)
		TOML_NODISCARD
		TOML_EXPORTED_MEMBER_FUNCTION
		node_view<const node> at_path(std::wstring_view path) const;

#endif // TOML_ENABLE_WINDOWS_COMPAT

		/// @}
	};
}
TOML_NAMESPACE_END;

/// \cond
TOML_IMPL_NAMESPACE_START
{
	TOML_PURE_GETTER
	TOML_EXPORTED_FREE_FUNCTION
	bool node_deep_equality(const node*, const node*) noexcept;
}
TOML_IMPL_NAMESPACE_END;
/// \endcond

#include "header_end.h"
