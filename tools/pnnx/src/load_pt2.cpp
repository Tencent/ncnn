// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_pt2.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <set>

#include "pt2_archive.h"
#include "pt2_graph_lowering.h"
#include "pt2_program.h"
#include "pt2_weights.h"

namespace pnnx {

static void skip_space(const char*& p)
{
    while (*p == ' ' || *p == '\t')
        p++;
}

static bool add_int64(int64_t a, int64_t b, int64_t& value)
{
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b))
        return false;
    value = a + b;
    return true;
}

static bool mul_int64(int64_t a, int64_t b, int64_t& value)
{
    if (a > 0 && ((b > 0 && a > INT64_MAX / b) || (b < 0 && b < INT64_MIN / a)))
        return false;
    if (a < 0 && ((b > 0 && a < INT64_MIN / b) || (b < 0 && a < INT64_MAX / b)))
        return false;
    value = a * b;
    return true;
}

static bool floor_div_int64(int64_t a, int64_t b, int64_t& value)
{
    if (b == 0 || (a == INT64_MIN && b == -1))
        return false;
    value = a / b;
    if (a % b != 0 && (a < 0) != (b < 0))
        value--;
    return true;
}

static bool parse_symbol(const char*& p, std::string& symbol)
{
    skip_space(p);
    if (*p++ != '\'')
        return false;
    const char* symbol_begin = p;
    while (*p && *p != '\'')
        p++;
    if (*p != '\'')
        return false;
    symbol.assign(symbol_begin, p++);
    if (symbol.empty())
        return false;

    int assumptions = 0;
    for (;;)
    {
        skip_space(p);
        if (*p == ')')
        {
            p++;
            return true;
        }
        if (*p++ != ',')
            return false;
        skip_space(p);
        const char* assumption_begin = p;
        while ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') || *p == '_')
            p++;
        const std::string assumption(assumption_begin, p);
        const int flag = assumption == "positive" ? 1 : assumption == "integer" ? 2 : 0;
        if (flag == 0 || (assumptions & flag))
            return false;
        assumptions |= flag;
        skip_space(p);
        if (*p++ != '=')
            return false;
        skip_space(p);
        const char* value_begin = p;
        while ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z'))
            p++;
        const std::string value(value_begin, p);
        if (value != "True" && value != "False")
            return false;
    }
}

static bool parse_sym_int(const char*& p, const std::map<std::string, int64_t>& symbols, const std::set<std::string>& runtime_symbols, int64_t& value, bool& dynamic)
{
    dynamic = false;
    skip_space(p);
    const char* name_begin = p;
    while ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') || *p == '_')
        p++;
    const std::string name(name_begin, p);
    if (*p++ != '(')
        return false;

    if (name == "Integer")
    {
        errno = 0;
        char* end;
        value = strtoll(p, &end, 10);
        if (errno || end == p)
            return false;
        p = end;
        skip_space(p);
        return *p++ == ')';
    }

    if (name == "Symbol")
    {
        std::string symbol;
        if (!parse_symbol(p, symbol))
            return false;
        std::map<std::string, int64_t>::const_iterator it = symbols.find(symbol);
        if (it != symbols.end())
        {
            value = it->second;
            return true;
        }
        dynamic = runtime_symbols.find(symbol) != runtime_symbols.end();
        value = 0;
        return dynamic;
    }

    if (name == "TruncToInt")
    {
        if (!parse_sym_int(p, symbols, runtime_symbols, value, dynamic))
            return false;
        skip_space(p);
        return *p++ == ')';
    }

    if (name == "FloorDiv")
    {
        int64_t a;
        int64_t b;
        bool dynamic_a;
        bool dynamic_b;
        if (!parse_sym_int(p, symbols, runtime_symbols, a, dynamic_a))
            return false;
        skip_space(p);
        if (*p++ != ',' || !parse_sym_int(p, symbols, runtime_symbols, b, dynamic_b))
            return false;
        skip_space(p);
        if (*p++ != ')')
            return false;
        dynamic = dynamic_a || dynamic_b;
        return dynamic || floor_div_int64(a, b, value);
    }

    if (name != "Add" && name != "Mul" && name != "Min" && name != "Max")
        return false;
    bool first = true;
    for (;;)
    {
        int64_t item;
        bool item_dynamic;
        if (!parse_sym_int(p, symbols, runtime_symbols, item, item_dynamic))
            return false;
        if (first)
            value = item;
        else if (!dynamic && !item_dynamic)
        {
            if (name == "Add" && !add_int64(value, item, value))
                return false;
            if (name == "Mul" && !mul_int64(value, item, value))
                return false;
            if (name == "Min" && item < value)
                value = item;
            if (name == "Max" && item > value)
                value = item;
        }
        dynamic |= item_dynamic;
        first = false;
        skip_space(p);
        if (*p == ')')
        {
            p++;
            return true;
        }
        if (*p++ != ',')
            return false;
    }
}

static bool evaluate_sym_int(const std::string& expression, const std::map<std::string, int64_t>& symbols, const std::set<std::string>& runtime_symbols, int64_t& value, bool& dynamic)
{
    const char* p = expression.c_str();
    if (!parse_sym_int(p, symbols, runtime_symbols, value, dynamic))
        return false;
    skip_space(p);
    return *p == '\0';
}

static bool get_symbol_name(const std::string& expression, std::string& name)
{
    const std::string prefix = "Symbol(";
    if (expression.compare(0, prefix.size(), prefix) != 0)
        return false;
    const char* p = expression.c_str() + prefix.size();
    if (!parse_symbol(p, name))
        return false;
    skip_space(p);
    return *p == '\0';
}

static bool parse_guard_value(const char*& p, const std::map<std::string, std::vector<int64_t> >& shapes, const std::map<std::string, int64_t>& symbols, int64_t& value);

static bool parse_guard_primary(const char*& p, const std::map<std::string, std::vector<int64_t> >& shapes, const std::map<std::string, int64_t>& symbols, int64_t& value)
{
    skip_space(p);
    if (strncmp(p, "L['", 3) == 0)
    {
        p += 3;
        const char* name_begin = p;
        while (*p && *p != '\'' && *p != '\\')
            p++;
        const std::string name(name_begin, p);
        if (name.empty() || strncmp(p, "'].size()[", 10) != 0)
            return false;
        p += 10;
        errno = 0;
        char* end;
        const long long index = strtoll(p, &end, 10);
        if (errno || end == p || index < 0 || *end != ']')
            return false;
        p = end + 1;
        std::map<std::string, std::vector<int64_t> >::const_iterator it = shapes.find(name);
        if (it == shapes.end() || (uint64_t)index >= it->second.size())
            return false;
        value = it->second[(size_t)index];
        return true;
    }

    if (strncmp(p, "max(", 4) == 0)
    {
        p += 3;
        skip_space(p);
        if (*p != '(')
            return false;
        p++;
        int64_t a;
        int64_t b;
        if (!parse_guard_value(p, shapes, symbols, a))
            return false;
        skip_space(p);
        if (*p != ',')
            return false;
        p++;
        if (!parse_guard_value(p, shapes, symbols, b))
            return false;
        skip_space(p);
        if (*p != ')')
            return false;
        p++;
        value = a > b ? a : b;
        return true;
    }

    if (*p == '(')
    {
        p++;
        if (!parse_guard_value(p, shapes, symbols, value))
            return false;
        skip_space(p);
        if (*p != ')')
            return false;
        p++;
        return true;
    }

    if ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') || *p == '_')
    {
        const char* begin = p++;
        while ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') || *p == '_')
            p++;
        std::map<std::string, int64_t>::const_iterator it = symbols.find(std::string(begin, p));
        if (it == symbols.end())
            return false;
        value = it->second;
        return true;
    }

    errno = 0;
    char* end;
    value = strtoll(p, &end, 10);
    if (errno || end == p)
        return false;
    p = end;
    return true;
}

static bool parse_guard_value(const char*& p, const std::map<std::string, std::vector<int64_t> >& shapes, const std::map<std::string, int64_t>& symbols, int64_t& value)
{
    if (!parse_guard_primary(p, shapes, symbols, value))
        return false;
    for (;;)
    {
        skip_space(p);
        if (p[0] != '/' || p[1] != '/')
            return true;
        p += 2;
        int64_t divisor;
        if (!parse_guard_primary(p, shapes, symbols, divisor) || !floor_div_int64(value, divisor, value))
            return false;
    }
}

static int evaluate_guard(const std::string& guard, const std::map<std::string, std::vector<int64_t> >& shapes, const std::map<std::string, int64_t>& symbols)
{
    const char* p = guard.c_str();
    skip_space(p);
    bool eq = false;
    if (strncmp(p, "Eq(", 3) == 0)
    {
        eq = true;
        p += 3;
    }

    int64_t a;
    int64_t b;
    if (!parse_guard_value(p, shapes, symbols, a))
        return -1;
    skip_space(p);

    if (eq)
    {
        if (*p++ != ',' || !parse_guard_value(p, shapes, symbols, b))
            return -1;
        skip_space(p);
        if (*p++ != ')')
            return -1;
        skip_space(p);
        if (*p)
            return -1;
        return a == b;
    }

    const char* op = p;
    if (p[0] && (p[0] == '=' || p[0] == '!' || p[0] == '<' || p[0] == '>') && p[1] == '=')
        p += 2;
    else if (p[0] == '<' || p[0] == '>')
        p++;
    else
        return -1;
    if (!parse_guard_value(p, shapes, symbols, b))
        return -1;
    skip_space(p);
    if (*p)
        return -1;

    if (op[0] == '=' && op[1] == '=') return a == b;
    if (op[0] == '!' && op[1] == '=') return a != b;
    if (op[0] == '<' && op[1] == '=') return a <= b;
    if (op[0] == '>' && op[1] == '=') return a >= b;
    if (op[0] == '<') return a < b;
    return a > b;
}

static const char* scalar_type_name(int type)
{
    static const char* names[] = {"", "u8", "i8", "i16", "i32", "i64", "f16", "f32", "f64", "c32", "c64", "c128", "bool", "bf16"};
    return type >= 1 && type <= 13 ? names[type] : "";
}

static int bind_input_shapes(const Pt2Program& program, const std::vector<std::vector<int64_t> >& input_shapes, const std::vector<std::string>& input_types,
                             const char* profile, std::map<std::string, int64_t>& symbols, std::string& error)
{
    if (input_shapes.size() != input_types.size())
    {
        error = std::string(profile) + " shape and type counts differ";
        return -1;
    }

    size_t input_index = 0;
    std::map<std::string, std::vector<int64_t> > shapes;
    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const Pt2InputSpec& spec = program.input_specs[i];
        if (spec.kind != Pt2InputSpec::UserInput)
            continue;
        if (spec.arg.type != Pt2Argument::Tensor)
        {
            error = "only tensor user inputs are supported";
            return -1;
        }
        if (!input_shapes.empty() && input_index >= input_shapes.size())
        {
            error = std::string(profile) + " has too few inputs";
            return -1;
        }

        const Pt2Tensor& tensor = program.tensors.at(spec.arg.s);
        if (!input_types.empty() && input_types[input_index] != scalar_type_name(tensor.dtype))
        {
            error = std::string(profile) + " input " + std::to_string(input_index) + " type must be " + scalar_type_name(tensor.dtype);
            return -1;
        }
        if (!input_shapes.empty() && input_shapes[input_index].size() != tensor.sizes.size())
        {
            error = std::string(profile) + " input " + std::to_string(input_index) + " rank mismatch";
            return -1;
        }
        std::vector<int64_t>& shape = shapes[spec.arg.s];
        for (size_t j = 0; j < tensor.sizes.size(); j++)
        {
            const Pt2SymInt& size = tensor.sizes[j];
            const int64_t dimension = input_shapes.empty() ? size.symbolic && size.has_hint ? size.hint : size.value : input_shapes[input_index][j];
            if (dimension < 0)
            {
                error = std::string(profile) + " input " + std::to_string(input_index) + " has a negative dimension";
                return -1;
            }
            shape.push_back(dimension);
            if (!size.symbolic)
            {
                if (dimension != size.value)
                {
                    error = std::string(profile) + " input " + std::to_string(input_index) + " dimension " + std::to_string(j) + " must be " + std::to_string(size.value);
                    return -1;
                }
                continue;
            }
            std::string symbol;
            if (!size.has_hint || !get_symbol_name(size.expression, symbol))
            {
                error = "unsupported symbolic input dimension " + size.expression;
                return -1;
            }
            std::map<std::string, int64_t>::iterator it = symbols.find(symbol);
            if (it != symbols.end() && it->second != dimension)
            {
                error = std::string(profile) + " assigns conflicting values to " + symbol;
                return -1;
            }
            symbols[symbol] = dimension;
        }
        input_index++;
    }
    if (!input_shapes.empty() && input_index != input_shapes.size())
    {
        error = std::string(profile) + " has too many inputs";
        return -1;
    }

    for (std::map<std::string, int64_t>::const_iterator it = symbols.begin(); it != symbols.end(); ++it)
    {
        std::map<std::string, Pt2RangeConstraint>::const_iterator range = program.range_constraints.find(it->first);
        if (range == program.range_constraints.end())
            continue;
        if (it->second < range->second.min || it->second > range->second.max)
        {
            error = std::string(profile) + " value " + std::to_string(it->second) + " for " + it->first + " is outside the exported range";
            return -1;
        }
    }
    for (size_t i = 0; i < program.guards_code.size(); i++)
    {
        const int result = evaluate_guard(program.guards_code[i], shapes, symbols);
        if (result < 0)
        {
            error = "cannot evaluate runtime guard " + program.guards_code[i] + "; simplify the export constraints or provide explicit inputshape profiles";
            return -1;
        }
        if (result == 0)
        {
            error = std::string(profile) + " violates runtime guard " + program.guards_code[i];
            return -1;
        }
    }
    return 0;
}

static int specialize_shapes(Pt2Program& program,
                             const std::vector<std::vector<int64_t> >& input_shapes, const std::vector<std::string>& input_types,
                             const std::vector<std::vector<int64_t> >& input_shapes2, const std::vector<std::string>& input_types2,
                             std::string& error)
{
    std::map<std::string, int64_t> symbols;
    if (bind_input_shapes(program, input_shapes, input_types, "inputshape", symbols, error) != 0)
        return -1;
    std::map<std::string, int64_t> symbols2;
    if (!input_shapes2.empty())
    {
        if (bind_input_shapes(program, input_shapes2, input_types2, "inputshape2", symbols2, error) != 0)
            return -1;
    }

    std::set<std::string> runtime_symbols;
    for (std::map<std::string, std::string>::const_iterator it = program.sym_floats.begin(); it != program.sym_floats.end(); ++it)
    {
        std::string symbol;
        if (get_symbol_name(it->second, symbol))
            runtime_symbols.insert(symbol);
    }

    for (std::map<std::string, Pt2Tensor>::iterator it = program.tensors.begin(); it != program.tensors.end(); ++it)
    {
        for (size_t i = 0; i < it->second.sizes.size(); i++)
        {
            Pt2SymInt& size = it->second.sizes[i];
            if (!size.symbolic)
                continue;
            int64_t value;
            bool dynamic;
            if (!evaluate_sym_int(size.expression, symbols, runtime_symbols, value, dynamic))
            {
                error = "unsupported symbolic expression " + size.expression;
                return -1;
            }
            if (dynamic)
            {
                size.symbolic = false;
                size.value = -1;
                continue;
            }
            if (value < 0)
            {
                error = "symbolic expression evaluates to a negative dimension " + size.expression;
                return -1;
            }
            size.value = value;
            if (!input_shapes2.empty())
            {
                int64_t value2;
                bool dynamic2;
                if (!evaluate_sym_int(size.expression, symbols2, runtime_symbols, value2, dynamic2) || dynamic2)
                {
                    error = "unsupported symbolic expression " + size.expression;
                    return -1;
                }
                if (value2 < 0)
                {
                    error = "symbolic expression evaluates to a negative dimension " + size.expression;
                    return -1;
                }
                if (value != value2)
                    size.value = -1;
            }
            size.symbolic = false;
        }
    }
    return 0;
}

int load_pt2(const std::string& path, Graph& graph,
             const std::vector<std::vector<int64_t> >& input_shapes, const std::vector<std::string>& input_types,
             const std::vector<std::vector<int64_t> >& input_shapes2, const std::vector<std::string>& input_types2)
{
    Pt2ArchiveReader archive;
    if (archive.open(path) != 0)
    {
        fprintf(stderr, "load pt2 archive failed: %s\n", archive.error.c_str());
        return -1;
    }

    Pt2Program program;
    if (load_pt2_program(archive, program) != 0)
    {
        fprintf(stderr, "load pt2 program failed: %s\n", program.error.c_str());
        return -1;
    }

    fprintf(stderr, "pt2 container=%s archive_version=%s", archive.container_kind == Pt2ContainerArchive ? "archive" : "legacy-exported-program", archive.archive_version.c_str());
    fprintf(stderr, " schema=%d.%d opset=", program.schema_major, program.schema_minor);
    for (std::map<std::string, int>::const_iterator it = program.opset_versions.begin(); it != program.opset_versions.end(); ++it)
    {
        if (it != program.opset_versions.begin())
            fprintf(stderr, ",");
        fprintf(stderr, "%s:%d", it->first.c_str(), it->second);
    }
    fprintf(stderr, " producer=torch%s%s\n", program.torch_version.empty() ? "" : "-", program.torch_version.c_str());

    std::string error;
    if (specialize_shapes(program, input_shapes, input_types, input_shapes2, input_types2, error) != 0)
    {
        fprintf(stderr, "specialize pt2 shapes failed: %s\n", error.c_str());
        return -1;
    }

    Pt2Weights weights;
    if (load_pt2_weights(archive, program, weights) != 0)
    {
        fprintf(stderr, "load pt2 weights failed: %s\n", weights.error.c_str());
        return -1;
    }

    if (lower_pt2_graph(program, weights, graph, error) != 0)
    {
        fprintf(stderr, "lower pt2 graph failed: %s\n", error.c_str());
        return -1;
    }

    return 0;
}

} // namespace pnnx
