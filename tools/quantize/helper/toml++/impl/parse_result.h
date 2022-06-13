//# This file is a part of toml++ and is subject to the the terms of the MIT license.
//# Copyright (c) Mark Gillard <mark.gillard@outlook.com.au>
//# See https://github.com/marzer/tomlplusplus/blob/master/LICENSE for the full license text.
// SPDX-License-Identifier: MIT
#pragma once

#include "preprocessor.h"
#if defined(DOXYGEN) || (TOML_ENABLE_PARSER && !TOML_EXCEPTIONS)

#include "table.h"
#include "parse_error.h"
#include "header_start.h"

TOML_NAMESPACE_START
{
	TOML_ABI_NAMESPACE_START(noex);

	/// \brief	The result of a parsing operation.
	///
	/// \availability <strong>This type only exists when exceptions are disabled.</strong>
	/// 		 Otherwise parse_result is just an alias for toml::table: \cpp
	/// #if TOML_EXCEPTIONS
	///		using parse_result = table;
	/// #else
	///		class parse_result { // ...
	///	#endif
	/// \ecpp
	///
	/// \detail A parse_result is effectively a discriminated union containing either a toml::table
	/// 		or a toml::parse_error. Most member functions assume a particular one of these two states,
	/// 		and calling them when in the wrong state will cause errors (e.g. attempting to access the
	/// 		error object when parsing was successful). \cpp
	/// toml::parse_result result = toml::parse_file("config.toml");
	/// if (result)
	///		do_stuff_with_a_table(result); //implicitly converts to table&
	///	else
	///		std::cerr << "Parse failed:\n"sv << result.error() << "\n";
	/// \ecpp
	///
	/// \out
	/// example output:
	///
	/// Parse failed:
	/// Encountered unexpected character while parsing boolean; expected 'true', saw 'trU'
	///		(error occurred at line 1, column 13 of 'config.toml')
	/// \eout
	///
	/// Getting node_views (`operator[]`, `at_path()`) and using the iterator accessor functions (`begin()`, `end()` etc.) are
	/// unconditionally safe; when parsing fails these just return 'empty' values. A ranged-for loop on a failed
	/// parse_result is also safe since `begin()` and `end()` return the same iterator and will not lead to any
	/// dereferences and iterations.
	class parse_result
	{
	  private:
		struct storage_t
		{
			static constexpr size_t size_ =
				(sizeof(toml::table) < sizeof(parse_error) ? sizeof(parse_error) : sizeof(toml::table));
			static constexpr size_t align_ =
				(alignof(toml::table) < alignof(parse_error) ? alignof(parse_error) : alignof(toml::table));

			alignas(align_) unsigned char bytes[size_];
		};

		mutable storage_t storage_;
		bool err_;

		template <typename Type>
		TOML_NODISCARD
		TOML_ALWAYS_INLINE
		static Type* get_as(storage_t& s) noexcept
		{
			return TOML_LAUNDER(reinterpret_cast<Type*>(s.bytes));
		}

		void destroy() noexcept
		{
			if (err_)
				get_as<parse_error>(storage_)->~parse_error();
			else
				get_as<toml::table>(storage_)->~table();
		}

	  public:
		/// \brief Default constructs an 'error' result.
		TOML_NODISCARD_CTOR
		parse_result() noexcept //
			: err_{ true }
		{
			::new (static_cast<void*>(storage_.bytes)) parse_error{ std::string{}, source_region{} };
		}

		TOML_NODISCARD_CTOR
		explicit parse_result(toml::table&& tbl) noexcept //
			: err_{ false }
		{
			::new (static_cast<void*>(storage_.bytes)) toml::table{ std::move(tbl) };
		}

		TOML_NODISCARD_CTOR
		explicit parse_result(parse_error&& err) noexcept //
			: err_{ true }
		{
			::new (static_cast<void*>(storage_.bytes)) parse_error{ std::move(err) };
		}

		/// \brief	Move constructor.
		TOML_NODISCARD_CTOR
		parse_result(parse_result&& res) noexcept //
			: err_{ res.err_ }
		{
			if (err_)
				::new (static_cast<void*>(storage_.bytes)) parse_error{ std::move(res).error() };
			else
				::new (static_cast<void*>(storage_.bytes)) toml::table{ std::move(res).table() };
		}

		/// \brief	Move-assignment operator.
		parse_result& operator=(parse_result&& rhs) noexcept
		{
			if (err_ != rhs.err_)
			{
				destroy();
				err_ = rhs.err_;
				if (err_)
					::new (static_cast<void*>(storage_.bytes)) parse_error{ std::move(rhs).error() };
				else
					::new (static_cast<void*>(storage_.bytes)) toml::table{ std::move(rhs).table() };
			}
			else
			{
				if (err_)
					error() = std::move(rhs).error();
				else
					table() = std::move(rhs).table();
			}
			return *this;
		}

		/// \brief	Destructor.
		~parse_result() noexcept
		{
			destroy();
		}

		/// \name Result state
		/// @{

		/// \brief	Returns true if parsing succeeeded.
		TOML_NODISCARD
		bool succeeded() const noexcept
		{
			return !err_;
		}

		/// \brief	Returns true if parsing failed.
		TOML_NODISCARD
		bool failed() const noexcept
		{
			return err_;
		}

		/// \brief	Returns true if parsing succeeded.
		TOML_NODISCARD
		explicit operator bool() const noexcept
		{
			return !err_;
		}

		/// @}

		/// \name Successful parses
		/// @{

		/// \brief	Returns the internal toml::table.
		TOML_NODISCARD
		toml::table& table() & noexcept
		{
			TOML_ASSERT_ASSUME(!err_);
			return *get_as<toml::table>(storage_);
		}

		/// \brief	Returns the internal toml::table (rvalue overload).
		TOML_NODISCARD
		toml::table&& table() && noexcept
		{
			TOML_ASSERT_ASSUME(!err_);
			return static_cast<toml::table&&>(*get_as<toml::table>(storage_));
		}

		/// \brief	Returns the internal toml::table (const lvalue overload).
		TOML_NODISCARD
		const toml::table& table() const& noexcept
		{
			TOML_ASSERT_ASSUME(!err_);
			return *get_as<const toml::table>(storage_);
		}

		/// \brief	Returns the internal toml::table.
		TOML_NODISCARD
		/* implicit */ operator toml::table&() noexcept
		{
			return table();
		}

		/// \brief	Returns the internal toml::table (rvalue overload).
		TOML_NODISCARD
		/* implicit */ operator toml::table&&() noexcept
		{
			return std::move(table());
		}

		/// \brief	Returns the internal toml::table (const lvalue overload).
		TOML_NODISCARD
		/* implicit */ operator const toml::table&() const noexcept
		{
			return table();
		}

		/// @}

		/// \name Failed parses
		/// @{

		/// \brief	Returns the internal toml::parse_error.
		TOML_NODISCARD
		parse_error& error() & noexcept
		{
			TOML_ASSERT_ASSUME(err_);
			return *get_as<parse_error>(storage_);
		}

		/// \brief	Returns the internal toml::parse_error (rvalue overload).
		TOML_NODISCARD
		parse_error&& error() && noexcept
		{
			TOML_ASSERT_ASSUME(err_);
			return static_cast<parse_error&&>(*get_as<parse_error>(storage_));
		}

		/// \brief	Returns the internal toml::parse_error (const lvalue overload).
		TOML_NODISCARD
		const parse_error& error() const& noexcept
		{
			TOML_ASSERT_ASSUME(err_);
			return *get_as<const parse_error>(storage_);
		}

		/// \brief	Returns the internal toml::parse_error.
		TOML_NODISCARD
		explicit operator parse_error&() noexcept
		{
			return error();
		}

		/// \brief	Returns the internal toml::parse_error (rvalue overload).
		TOML_NODISCARD
		explicit operator parse_error&&() noexcept
		{
			return std::move(error());
		}

		/// \brief	Returns the internal toml::parse_error (const lvalue overload).
		TOML_NODISCARD
		explicit operator const parse_error&() const noexcept
		{
			return error();
		}

		/// @}

		/// \name Iteration
		/// @{

		/// \brief A BidirectionalIterator for iterating over key-value pairs in a wrapped toml::table.
		using iterator = table_iterator;

		/// \brief A BidirectionalIterator for iterating over const key-value pairs in a wrapped toml::table.
		using const_iterator = const_table_iterator;

		/// \brief	Returns an iterator to the first key-value pair in the wrapped table.
		/// \remarks Always returns the same value as #end() if parsing failed.
		TOML_NODISCARD
		table_iterator begin() noexcept
		{
			return err_ ? table_iterator{} : table().begin();
		}

		/// \brief	Returns an iterator to the first key-value pair in the wrapped table.
		/// \remarks Always returns the same value as #end() if parsing failed.
		TOML_NODISCARD
		const_table_iterator begin() const noexcept
		{
			return err_ ? const_table_iterator{} : table().begin();
		}

		/// \brief	Returns an iterator to the first key-value pair in the wrapped table.
		/// \remarks Always returns the same value as #cend() if parsing failed.
		TOML_NODISCARD
		const_table_iterator cbegin() const noexcept
		{
			return err_ ? const_table_iterator{} : table().cbegin();
		}

		/// \brief	Returns an iterator to one-past-the-last key-value pair in the wrapped table.
		TOML_NODISCARD
		table_iterator end() noexcept
		{
			return err_ ? table_iterator{} : table().end();
		}

		/// \brief	Returns an iterator to one-past-the-last key-value pair in the wrapped table.
		TOML_NODISCARD
		const_table_iterator end() const noexcept
		{
			return err_ ? const_table_iterator{} : table().end();
		}

		/// \brief	Returns an iterator to one-past-the-last key-value pair in the wrapped table.
		TOML_NODISCARD
		const_table_iterator cend() const noexcept
		{
			return err_ ? const_table_iterator{} : table().cend();
		}

		/// @}

		/// \name Node views
		/// @{

		/// \brief	Gets a node_view for the selected key-value pair in the wrapped table.
		///
		/// \param 	key The key used for the lookup.
		///
		/// \returns	A view of the value at the given key if parsing was successful and a matching key existed,
		/// 			or an empty node view.
		///
		/// \see toml::node_view
		TOML_NODISCARD
		node_view<node> operator[](std::string_view key) noexcept
		{
			return err_ ? node_view<node>{} : table()[key];
		}

		/// \brief	Gets a node_view for the selected key-value pair in the wrapped table (const overload).
		///
		/// \param 	key The key used for the lookup.
		///
		/// \returns	A view of the value at the given key if parsing was successful and a matching key existed,
		/// 			or an empty node view.
		///
		/// \see toml::node_view
		TOML_NODISCARD
		node_view<const node> operator[](std::string_view key) const noexcept
		{
			return err_ ? node_view<const node>{} : table()[key];
		}

		/// \brief Returns a view of the subnode matching a fully-qualified "TOML path".
		///
		/// \see #toml::node::at_path(std::string_view)
		TOML_NODISCARD
		node_view<node> at_path(std::string_view path) noexcept
		{
			return err_ ? node_view<node>{} : table().at_path(path);
		}

		/// \brief Returns a const view of the subnode matching a fully-qualified "TOML path".
		///
		/// \see #toml::node::at_path(std::string_view)
		TOML_NODISCARD
		node_view<const node> at_path(std::string_view path) const noexcept
		{
			return err_ ? node_view<const node>{} : table().at_path(path);
		}

#if TOML_ENABLE_WINDOWS_COMPAT

		/// \brief	Gets a node_view for the selected key-value pair in the wrapped table.
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \param 	key The key used for the lookup.
		///
		/// \returns	A view of the value at the given key if parsing was successful and a matching key existed,
		/// 			or an empty node view.
		///
		/// \see toml::node_view
		TOML_NODISCARD
		node_view<node> operator[](std::wstring_view key) noexcept
		{
			return err_ ? node_view<node>{} : table()[key];
		}

		/// \brief	Gets a node_view for the selected key-value pair in the wrapped table (const overload).
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \param 	key The key used for the lookup.
		///
		/// \returns	A view of the value at the given key if parsing was successful and a matching key existed,
		/// 			or an empty node view.
		///
		/// \see toml::node_view
		TOML_NODISCARD
		node_view<const node> operator[](std::wstring_view key) const noexcept
		{
			return err_ ? node_view<const node>{} : table()[key];
		}

		/// \brief Returns a view of the subnode matching a fully-qualified "TOML path".
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \see #toml::node::at_path(std::string_view)
		TOML_NODISCARD
		node_view<node> at_path(std::wstring_view path) noexcept
		{
			return err_ ? node_view<node>{} : table().at_path(path);
		}

		/// \brief Returns a const view of the subnode matching a fully-qualified "TOML path".
		///
		/// \availability This overload is only available when #TOML_ENABLE_WINDOWS_COMPAT is enabled.
		///
		/// \see #toml::node::at_path(std::string_view)
		TOML_NODISCARD
		node_view<const node> at_path(std::wstring_view path) const noexcept
		{
			return err_ ? node_view<const node>{} : table().at_path(path);
		}

#endif // TOML_ENABLE_WINDOWS_COMPAT

		/// @}

#if TOML_ENABLE_FORMATTERS

		/// \brief Prints the held error or table object out to a text stream.
		///
		/// \availability This operator is only available when #TOML_ENABLE_FORMATTERS is enabled.
		friend std::ostream& operator<<(std::ostream& os, const parse_result& result)
		{
			return result.err_ ? (os << result.error()) : (os << result.table());
		}

#endif
	};

	TOML_ABI_NAMESPACE_END;
}
TOML_NAMESPACE_END;

#include "header_end.h"
#endif // TOML_ENABLE_PARSER && !TOML_EXCEPTIONS
