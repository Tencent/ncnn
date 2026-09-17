#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the offline ATen argument default table."""

import argparse
import datetime
import json
import sys
import zipfile
from pathlib import Path

import torch

# Keep these values synchronized with Pt2DefaultType.
T_NO_DEFAULT = -1
T_NONE = 0
T_INT = 1
T_FLOAT = 2
T_BOOL = 3
T_STRING = 4
T_INTS = 5
T_FLOATS = 6
T_STRINGS = 7
T_DEVICE = 8
T_UNSUPPORTED = 9

TYPE_NAMES = {
    T_NO_DEFAULT: "NO_DEFAULT",
    T_NONE: "NONE",
    T_INT: "INT",
    T_FLOAT: "FLOAT",
    T_BOOL: "BOOL",
    T_STRING: "STRING",
    T_INTS: "INTS",
    T_FLOATS: "FLOATS",
    T_STRINGS: "STRINGS",
    T_DEVICE: "DEVICE",
    T_UNSUPPORTED: "UNSUPPORTED",
}


class UnsupportedDefault(Exception):
    pass


def _float_repr(v):
    return repr(float(v))


def encode_default(dv):
    if dv is None:
        return T_NONE, ""
    if isinstance(dv, bool):
        return T_BOOL, "1" if dv else "0"
    if isinstance(dv, int):
        return T_INT, str(dv)
    if isinstance(dv, float):
        return T_FLOAT, _float_repr(dv)
    if isinstance(dv, str):
        return T_STRING, dv
    if isinstance(dv, torch.device):
        return T_DEVICE, str(dv) if dv.type else ""
    if isinstance(dv, list):
        if not dv:
            return T_INTS, ""
        if all(isinstance(x, bool) for x in dv):
            raise UnsupportedDefault("bool list(pnnx Parameter 无 bool 列表表达)")
        if all(isinstance(x, int) for x in dv):
            return T_INTS, ",".join(str(x) for x in dv)
        if all(isinstance(x, float) for x in dv):
            return T_FLOATS, ",".join(_float_repr(x) for x in dv)
        if all(isinstance(x, str) for x in dv):
            if any("," in x for x in dv):
                raise UnsupportedDefault("string list 元素含逗号,平铺编码有歧义")
            return T_STRINGS, ",".join(dv)
        raise UnsupportedDefault(f"列表元素类型混合或张量: {dv!r}")
    raise UnsupportedDefault(f"类型 {type(dv).__name__}: {dv!r}")


def collect_ops_from_pt2(path):
    ops = set()
    with zipfile.ZipFile(path) as zf:
        entries = [n for n in zf.namelist() if n.endswith("models/model.json")]
        if not entries:
            print(f"  SKIP(非 pt2): {path}", file=sys.stderr)
            return ops
        data = json.loads(zf.read(entries[0]))
    for node in data.get("graph_module", {}).get("graph", {}).get("nodes", []):
        target = node.get("target", "")
        if not target.startswith("torch.ops."):
            continue
        rest = target[len("torch.ops."):]
        ns, sep, tail = rest.partition(".")
        if not sep:
            continue
        ops.add(ns + "::" + tail)
    return ops


def scan_dirs(dirs):
    ops = set()
    for d in dirs:
        pt2s = sorted(Path(d).rglob("*.pt2"))
        print(f"scan {d}: {len(pt2s)} 个 .pt2", file=sys.stderr)
        for p in pt2s:
            try:
                ops |= collect_ops_from_pt2(p)
            except Exception as e:  # noqa: BLE001
                print(f"  WARN 读取失败 {p}: {e}", file=sys.stderr)
    return ops


def build_schema_index():
    index = {}
    for s in torch._C._jit_get_all_schemas():
        overload = s.overload_name or "default"
        index[f"{s.name}.{overload}"] = s
    return index


def resolve_op_names(requested, schema_index):
    resolved = []
    for name in requested:
        name = name.strip()
        if not name:
            continue
        if name in schema_index:
            resolved.append(name)
            continue
        cands = [k for k in schema_index if k.startswith(name + ".")]
        if not cands:
            print(f"WARN schema 未收录: {name}", file=sys.stderr)
            continue
        if f"{name}.default" in cands:
            resolved.append(f"{name}.default")
        else:
            print(f"WARN {name} 无 .default overload,候选 {cands} 未收录(请用全名显式指定)",
                  file=sys.stderr)
    return resolved


def escape_cpp_string(s):
    out = []
    for ch in s:
        if ch in ('\\', '"'):
            out.append("\\" + ch)
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.append("\\%03o" % ord(ch))
    return "".join(out)


def mangle_op_name(full_name):
    return full_name.replace("::", "_").replace(".", "_")


def dump_op(full_name, schema):
    rows = []
    for arg in schema.arguments:
        if arg.has_default_value():
            try:
                t, v = encode_default(arg.default_value)
            except UnsupportedDefault as e:
                print(f"  WARN {full_name}.{arg.name}: 默认值不可编码({e}),降级 UNSUPPORTED",
                      file=sys.stderr)
                t, v = T_UNSUPPORTED, ""
        else:
            t, v = T_NO_DEFAULT, ""
        rows.append((arg.name, t, v))
    return rows


HEADER_TEMPLATE = """\
// Copyright {year} Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// Static ATen argument default table. Generated offline; do not edit manually.
//
// torch.export omits graph inputs that equal operator defaults (such as cat
// dim=0, flatten end_dim=-1, and conv2d dilation/groups). The PT2 builder
// restores omitted inputs to the complete schema form so PT2 graphs match
// TorchScript graphs and existing pass_level2 patterns can be reused.
//
// Regenerate: {cmd}
// Source: torch._C._jit_get_all_schemas() from torch {torch_version} ({count_schemas} schemas)
// Generated: {date}
// Operators: {count_ops}
//
// Value encoding (type tag and string value):
//   NO_DEFAULT=-1  Required argument without a default; preserves argument order.
//   NONE=0         ""
//   INT=1          Decimal integer.
//   FLOAT=2        Value accepted by strtod, including inf, -inf, and nan.
//   BOOL=3         "0"/"1"
//   STRING=4       Verbatim text.
//   INTS/FLOATS/STRINGS=5/6/7  Comma-separated flat values. An empty value is
//                  an empty list, represented as type 0 (None) by the builder.
//   DEVICE=8       "" is None; otherwise "cpu" or "cuda:0", stored as STRING.
//   UNSUPPORTED=9  Defaults not expressible by the builder, such as bool lists,
//                  nested lists, and Tensor values; never materialized.
//
// Coverage grows with the test corpus. For unlisted operators, the builder
// preserves the torch.export form and emits a missing-default warning.

#ifndef PNNX_ATEN_DEFAULTS_TABLE_H
#define PNNX_ATEN_DEFAULTS_TABLE_H

#include <stddef.h>
#include <string.h>

namespace pnnx {{

enum Pt2DefaultType
{{
    PT2_D_NO_DEFAULT = -1,
    PT2_D_NONE = 0,
    PT2_D_INT = 1,
    PT2_D_FLOAT = 2,
    PT2_D_BOOL = 3,
    PT2_D_STRING = 4,
    PT2_D_INTS = 5,
    PT2_D_FLOATS = 6,
    PT2_D_STRINGS = 7,
    PT2_D_DEVICE = 8,
    PT2_D_UNSUPPORTED = 9
}};

struct Pt2ArgDefault
{{
    const char* name;
    Pt2DefaultType type;
    const char* value;
}};

struct Pt2DefaultsEntry
{{
    const char* op; // Full name with overload, such as "aten::conv2d.default".
    const Pt2ArgDefault* args;
    size_t arg_count;
}};

// Finds argument defaults by complete PT2 target name, such as
// "aten::flatten.using_ints". Returns 0 when the target is not listed.
inline const Pt2DefaultsEntry* find_pt2_aten_defaults(const char* op)
{{
{entries}
    return 0;
}}

}} // namespace pnnx

#endif // PNNX_ATEN_DEFAULTS_TABLE_H
"""


def generate_header(op_rows, cmd, out_path):
    year = datetime.date.today().year
    count_schemas = len(torch._C._jit_get_all_schemas())

    entries_lines = []
    for full_name, rows in op_rows:
        mangled = mangle_op_name(full_name)
        entries_lines.append(f"    static const Pt2ArgDefault args_{mangled}[] = {{")
        for name, t, v in rows:
            entries_lines.append("        {\"%s\", PT2_D_%s, \"%s\"}," % (escape_cpp_string(name), TYPE_NAMES[t], escape_cpp_string(v)))
        entries_lines.append("    };")
        entries_lines.append(
            "    static const Pt2DefaultsEntry entry_%s = {\"%s\", args_%s, %d};"
            % (mangled, escape_cpp_string(full_name), mangled, len(rows))
        )
        entries_lines.append("    if (strcmp(op, entry_%s.op) == 0) return &entry_%s;" % (mangled, mangled))
        entries_lines.append("")

    header = HEADER_TEMPLATE.format(
        year=year,
        cmd=cmd,
        torch_version=torch.__version__,
        count_schemas=count_schemas,
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        count_ops=len(op_rows),
        entries="\n".join(entries_lines).rstrip("\n"),
    )

    Path(out_path).write_text(header, encoding="utf-8", newline="\n")
    print(f"生成 {out_path}: {len(op_rows)} 个算子", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", nargs="*", default=[], help="递归扫描目录里的 *.pt2,收录其中出现过的 aten op")
    ap.add_argument("--ops", nargs="*", default=[], help="显式追加 op(全名或省略 overload)")
    ap.add_argument("--all", action="store_true", help="收录全部 aten schema(含所有默认值 op,表很大)")
    ap.add_argument("--out", default="src/aten_defaults_table.h", help="输出头文件路径")
    args = ap.parse_args()

    if not (args.scan or args.ops or args.all):
        ap.error("至少提供 --scan / --ops / --all 之一")

    schema_index = build_schema_index()

    requested = set(args.ops)
    if args.scan:
        scanned = scan_dirs(args.scan)
        print(f"scan 命中 {len(scanned)} 个 op", file=sys.stderr)
        requested |= scanned
    if args.all:
        requested |= {k for k in schema_index if k.startswith("aten::")}

    op_names = resolve_op_names(sorted(requested), schema_index)

    op_names = sorted(set(op_names))

    op_rows = []
    for full in op_names:
        rows = dump_op(full, schema_index[full])
        op_rows.append((full, rows))

    cmd = "python scripts/dump_aten_defaults.py"
    if args.scan:
        cmd += " --scan " + " ".join(args.scan)
    if args.ops:
        cmd += " --ops " + " ".join(sorted(set(args.ops)))
    if args.all:
        cmd += " --all"

    generate_header(op_rows, cmd, args.out)


if __name__ == "__main__":
    main()
